########################################################################################################################
# Copyright 2021 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
# This file is part of OpenCMP.                                                                                        #
#                                                                                                                      #
# OpenCMP is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any     #
# later version.                                                                                                       #
#                                                                                                                      #
# OpenCMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more  #
# details.                                                                                                             #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCMP. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

"""Geometry-independent wall functions for the k-epsilon turbulence model."""

import numpy as np
import ngsolve as ngs


class KEpsilonWallFunction:
    """Wall-layer eddy viscosity and dissipation for high-Re k-epsilon.

    The wall layer is found from mesh topology: cells owning a facet on
    ``wall_boundary``, plus one face-connected neighbour layer. ``eval_nu_t``
    puts those cells on the wall law and every other cell on bulk
    ``C_mu k^2/epsilon``.

    The friction velocity ``u_tau = C_mu**0.25 * sqrt(k)`` is a P0 projection
    on the wall-facet owners, extended to their neighbours by arithmetic mean.
    ``update`` refreshes it and must run once per Picard iteration, before
    ``eval_nu_t``.

    Wall distance comes from a regularized-Eikonal solve: the continuous field
    for coefficients integrated across a cell (``y_plus_field``,
    ``eval_nu_wall``), and its cell average for per-cell quantities
    (``y_plus_cell``, ``epsilon_wall_cell``). Turbulence production is not
    handled here -- ``KEpsilonINS`` owns it.
    """

    #: y+ below which the viscous sublayer is assumed (no wall eddy viscosity).
    YPLUS_VISCOUS = 11.25
    #: y+ above which the log law stops applying and bulk k-epsilon takes over.
    YPLUS_LOG_MAX = 200.0

    def __init__(self, mesh: ngs.comp.Mesh, nu: float, C_mu: float, kappa: float,
                 E_log: float = 9.8, wall_boundary: str = "wall",
                 dist_order: int = 2, dist_relax: float = 0.1) -> None:
        self.mesh = mesh
        self.nu = nu                      # laminar kinematic viscosity
        self.C_mu = C_mu
        self.kappa = kappa
        self.E_log = E_log                # log-law roughness constant E
        self.wall_boundary = wall_boundary
        self.h = ngs.specialcf.mesh_size

        # Piecewise-constant space shared by the masks, the cell distance and u_tau.
        self._fes0 = ngs.L2(mesh, order=0)

        self._mark_wall_cells()

        self._dist_gf = self._compute_distance_field(dist_order, dist_relax)
        self._dist_cell = ngs.GridFunction(self._fes0)
        self._dist_cell.Set(self._dist_gf)
        # y+ and epsilon_wall_cell both divide by this; guard against a Newton
        # undershoot on a degenerate cell producing a negative distance.
        distance = self._dist_cell.vec.FV().NumPy()
        distance[:] = np.maximum(distance, 1e-12)

        # Zero until the first update(); eval_nu_t is then simply bulk everywhere.
        self.u_tau_cell = ngs.GridFunction(self._fes0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _mark_wall_cells(self) -> None:
        """Find the wall layer from mesh topology.

        Sets ``_wall_measure``, ``_marked`` / ``wall_facet_cell_mask`` (the
        wall-facet owners), and ``mask`` (those owners plus one neighbour
        layer -- the region the wall law is applied on).
        """
        # Assumes a simplicial mesh (one L2(0) DOF per cell). Quads/hexes parse
        # but the resulting layer is untested, so refuse rather than go silently wrong.
        element_types = {element.type for element in self.mesh.Elements(ngs.VOL)}
        if not element_types <= {ngs.ET.TRIG, ngs.ET.TET}:
            raise NotImplementedError(
                'KEpsilonWallFunction supports simplicial meshes only (TRIG in 2D, '
                f'TET in 3D); this mesh contains {sorted(t.name for t in element_types)}.')

        # An L2 basis function has no boundary trace, so plain ds() assembles to
        # zero; skeleton=True evaluates it on the facet instead.
        lf = ngs.LinearForm(self._fes0)
        lf += self._fes0.TestFunction() * ngs.ds(
            definedon=self.mesh.Boundaries(self.wall_boundary), skeleton=True)
        lf.Assemble()
        self._wall_measure = np.array(lf.vec).ravel()

        # Wall-facet owners: the only cells where u_tau can be evaluated directly,
        # and where the algebraic epsilon condition is anchored.
        self._marked = self._wall_measure > 0.0
        if not self._marked.any():
            raise ValueError(
                f"KEpsilonWallFunction: boundary marker '{self.wall_boundary}' has "
                f"no boundary elements. Available boundaries: "
                f"{self.mesh.GetBoundaries()}.")
        self.wall_facet_cell_mask = ngs.GridFunction(self._fes0)
        self.wall_facet_cell_mask.vec.FV().NumPy()[:] = self._marked.astype(float)

        # The extra neighbour layer suppresses bulk-model production on cells
        # immediately touching the wall layer across an interior facet.
        self.mask = ngs.GridFunction(self._fes0)
        self.mask.vec.FV().NumPy()[:] = self._expand_by_one_facet_layer(
            self._marked).astype(float)

    def _expand_by_one_facet_layer(self, wall_marked: np.ndarray) -> np.ndarray:
        """Add every volume cell sharing a facet with a wall-facet owner."""
        elements = list(self._fes0.Elements(ngs.VOL))
        wall_elements = {
            element.nr for element in elements
            if any(wall_marked[dof] for dof in element.dofs)
        }

        facet_to_elements = {}
        element_by_number = {element.nr: element for element in elements}
        self._element_dofs = {
            element.nr: tuple(element.dofs) for element in elements
        }
        for element in elements:
            for facet in element.facets:
                facet_to_elements.setdefault(facet.nr, set()).add(element.nr)

        expanded_elements = set(wall_elements)
        neighbour_sources = {}
        for element_number in wall_elements:
            for facet in element_by_number[element_number].facets:
                neighbours = facet_to_elements[facet.nr]
                expanded_elements.update(neighbours)
                for neighbour in neighbours - wall_elements:
                    neighbour_sources.setdefault(neighbour, set()).add(
                        element_number)
        self._wall_layer_sources = neighbour_sources

        expanded = np.zeros_like(wall_marked, dtype=bool)
        for element in elements:
            if element.nr in expanded_elements:
                expanded[list(element.dofs)] = True
        return expanded

    def _compute_distance_field(self, order: int, relax: float) -> ngs.GridFunction:
        """Distance to ``wall_boundary`` from a regularized Eikonal solve."""
        eps = relax * self.h
        fes = ngs.H1(self.mesh, order=order, dirichlet=self.wall_boundary)
        u, v = fes.TnT()
        y = ngs.GridFunction(fes)

        a = ngs.BilinearForm(fes)
        a += ngs.grad(u) * ngs.grad(v) * ngs.dx
        f = ngs.LinearForm(fes)
        f += 1.0 * v * ngs.dx
        a.Assemble()
        f.Assemble()
        y.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

        gu = ngs.grad(u)
        residual = ngs.BilinearForm(fes)
        residual += (
            ngs.sqrt(gu * gu + 1e-12) * v - v
            + eps * gu * ngs.grad(v)
        ) * ngs.dx
        ngs.solvers.Newton(residual, y, printing=False)
        return y

    # ------------------------------------------------------------------
    # Per-iteration update
    # ------------------------------------------------------------------

    def update(self, K) -> None:
        """Refresh the friction velocity from the current k iterate.

        Call once per Picard iteration, BEFORE ``eval_nu_t``.
        """
        projected = ngs.GridFunction(self._fes0)
        projected.Set(K)
        k_values = np.maximum(projected.vec.FV().NumPy(), 0.0)

        u_tau = np.zeros_like(k_values)
        u_tau[self._marked] = self.C_mu ** 0.25 * np.sqrt(k_values[self._marked])

        # A neighbour belongs to the same local wall-normal layer as its
        # wall-facet source cell(s), so inherit their u_tau instead of
        # recalculating it from the neighbour's independently varying k.
        for element_number, source_numbers in self._wall_layer_sources.items():
            inherited = float(np.mean([
                u_tau[self._element_dofs[source_number][0]]
                for source_number in source_numbers
            ]))
            u_tau[list(self._element_dofs[element_number])] = inherited

        self.u_tau_cell.vec.FV().NumPy()[:] = u_tau

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def near_wall_mask(self) -> ngs.GridFunction:
        """1 on wall-facet owners and their first face-connected neighbours."""
        return self.mask

    def wall_facet_mask(self) -> ngs.GridFunction:
        """1 only on cells owning a physical wall facet."""
        return self.wall_facet_cell_mask

    def wall_distance_cell(self) -> ngs.GridFunction:
        return self._dist_cell

    def wall_distance_field(self) -> ngs.GridFunction:
        """Continuous Eikonal wall-distance field."""
        return self._dist_gf

    def y_plus_cell(self, K=None) -> ngs.CoefficientFunction:
        """Cellwise y+, from the cell-averaged wall distance."""
        if K is not None:
            self.update(K)
        return self._dist_cell * self.u_tau_cell / self.nu

    def y_plus_field(self, K=None) -> ngs.CoefficientFunction:
        """Pointwise y+, from the continuous Eikonal distance.

        Use this wherever y+ feeds a coefficient the weak form integrates over a
        volume, so the result varies across the wall cell instead of being one
        number per cell.
        """
        if K is not None:
            self.update(K)
        return self._dist_gf * self.u_tau_cell / self.nu

    def epsilon_wall_cell(self, K) -> ngs.CoefficientFunction:
        """Dissipation in a wall-layer cell, switched at ``YPLUS_VISCOUS``:

        * log layer: ``C_mu**0.75 * k**1.5 / (kappa * y)``
        * viscous sublayer: ``2 * nu * k / y**2``

        The viscous branch is needed because production is zero below
        ``YPLUS_VISCOUS``; log-layer dissipation there suppresses k, which lowers
        y+, which keeps the cell below the switch -- an absorbing state.
        """
        k_nonnegative = ngs.IfPos(K, K, 0.0)
        log_layer = (self.C_mu ** 0.75 * k_nonnegative ** 1.5
                     / (self.kappa * self._dist_cell))
        viscous_sublayer = (2.0 * self.nu * k_nonnegative
                            / self._dist_cell ** 2)
        return ngs.IfPos(self.y_plus_cell(K) - self.YPLUS_VISCOUS,
                         log_layer, viscous_sublayer)

    # ------------------------------------------------------------------
    # Eddy viscosity
    # ------------------------------------------------------------------

    def _wall_law(self, yplus, K, epsilon) -> ngs.CoefficientFunction:
        """Blended wall eddy viscosity as a function of the y+ handed in::

            nu_t = 0                                     y+ <  11.25   (sublayer)
            nu_t = nu*(y+ / ((1/kappa)*ln(E*y+)) - 1)     y+ <  200     (log law)
            nu_t = C_mu*k^2/epsilon                       y+ >= 200     (bulk)
        """
        # ln(E*y+) vanishes at y+ = 1/E; pinning y+ to YPLUS_VISCOUS below the
        # sublayer threshold avoids that root (the clamp below then zeroes it).
        yplus_log = ngs.IfPos(yplus - self.YPLUS_VISCOUS, yplus, self.YPLUS_VISCOUS)
        u_plus = ngs.log(self.E_log * yplus_log) / self.kappa
        log_law = self.nu * (yplus_log / u_plus - 1.0)
        log_law = ngs.IfPos(log_law, log_law, 0.0)

        return ngs.IfPos(self.YPLUS_LOG_MAX - yplus,
                         log_law, self.C_mu * K ** 2 / epsilon)

    def eval_nu_wall(self, K, epsilon) -> ngs.CoefficientFunction:
        """Wall eddy viscosity as a profile across the near-wall cell.

        y+ comes from :meth:`y_plus_field`, so the coefficient varies within the
        cell rather than taking one value per cell.
        """
        return self._wall_law(self.y_plus_field(K), K, epsilon)

    def eval_nu_wall_cell(self, K, epsilon) -> ngs.CoefficientFunction:
        """Wall eddy viscosity at the cell-averaged distance, so it's nonzero at
        the wall trace. For scaling the Dirichlet penalty only; the physical
        viscosity profile is :meth:`eval_nu_wall`.
        """
        return self._wall_law(self.y_plus_cell(K), K, epsilon)

    def eval_nu_t(self, K, E) -> ngs.CoefficientFunction:
        """Wall law inside the mask, bulk ``C_mu k^2/epsilon`` outside it.

        Requires :meth:`update` to have run this iteration.
        """
        # Cell-based, not pointwise: the Eikonal distance is zero ON the wall, so
        # a pointwise wall law collapses there and jumps to bulk one cell in.
        nu_t_bulk = self.C_mu * K ** 2 / E
        nu_t = self.mask * self.eval_nu_wall(K, E) + (1 - self.mask) * nu_t_bulk
        return ngs.IfPos(nu_t, nu_t, 0)  # turbulent viscosity cannot be negative
