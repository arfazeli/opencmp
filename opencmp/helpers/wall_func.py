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

"""
Wall-function eddy-viscosity model for the k-epsilon two-fluid turbulence
closure.  Every physical constant and region/boundary name is passed in
through the constructor so the class carries no global state.
"""

import numpy as np
import ngsolve as ngs
from typing import Optional

from .ngsolve_ import construct_identity_mat


class WallFunc:
    """Law-of-the-wall blended turbulent viscosity.

    nu_t = bulk C_mu k^2/eps in the core region, replaced by a wall-function
    value in the near-wall region.  The wall-normal distance comes from a
    regularized-Eikonal solve; the wall-function value needs the per-x
    cross-section-averaged wall shear stress, refreshed each step via
    ``update_tau_avg``.
    """

    def __init__(self, mesh: ngs.comp.Mesh, mu: float, rho: float, nu: float,
                 C_mu: float, kappa: float, E: float,
                 cylinder_r: Optional[float] = None,
                 wall_boundary: str = "wall",
                 nearwall_region: str = "nearwall",
                 core_region_2d: str = "surface",
                 core_region_3d: str = "vol",
                 n_xbins: int = 100, dist_order: int = 2,
                 dist_relax: float = 0.1) -> None:
        self.mesh = mesh
        self.mu = mu
        self.rho = rho
        self.nu = nu                      # laminar kinematic viscosity nu_c
        self.C_mu = C_mu
        self.kappa = kappa
        self.E = E
        self.cylinder_r = cylinder_r
        self.wall_boundary = wall_boundary
        self.nearwall_region = nearwall_region
        self.core_region = core_region_2d if mesh.dim == 2 else core_region_3d
        self.n_xbins = n_xbins
        self.n = ngs.specialcf.normal(mesh.dim)
        self.h = ngs.specialcf.mesh_size
        self.IM = construct_identity_mat(mesh.dim)

        # Wall-distance field, solved once via the regularized Eikonal equation.
        self._dist_gf = self._compute_distance_field(dist_order, dist_relax)

        # Per-cell wall-normal distance (cell average of y_normal).
        self._dist_cell = ngs.GridFunction(ngs.L2(mesh, order=0))
        self._dist_cell.Set(self._dist_gf)

        # Domain bounds. cylinder_r remains accepted for compatibility with
        # the original pipe model, but arbitrary cross-sections use the mesh
        # bounds when no radius is supplied.
        points = np.array([v.point for v in mesh.vertices], dtype=float)
        xs = points[:, 0]
        self.x_min, self.x_max = float(xs.min()), float(xs.max())
        self._domain_min = tuple(points.min(axis=0))
        self._domain_max = tuple(points.max(axis=0))

        # tau_avg(x) placeholder, refreshed each step by update_tau_avg().
        self._tau_fes = ngs.H1(mesh, order=1)
        self.tau_avg_gf = ngs.GridFunction(self._tau_fes)
        self.tau_avg_gf.Set(ngs.CoefficientFunction(1e-12))

        # (surface-element nr, x-centroid) for all wall facets, computed once.
        self._wall_els = [
            (el.nr, float(np.mean([mesh.vertices[v.nr].point[0] for v in el.vertices])))
            for el in mesh.Elements(ngs.BND) if el.mat == wall_boundary
        ]
        if not self._wall_els:
            raise ValueError(
                f"WallFunc: boundary marker '{wall_boundary}' has no boundary elements.")

    def _compute_distance_field(self, order: int, relax: float) -> ngs.GridFunction:
        """Distance to the "wall" boundary via the regularized Eikonal equation
        |grad(y)| = 1 with y = 0 on the wall.  Poisson warm start, then Newton."""
        eps = relax * self.h
        fes = ngs.H1(self.mesh, order=order, dirichlet=self.wall_boundary)
        u, v = fes.TnT()
        y = ngs.GridFunction(fes)

        a = ngs.BilinearForm(fes)
        a += ngs.grad(u) * ngs.grad(v) * ngs.dx
        f = ngs.LinearForm(fes)
        f += 1.0 * v * ngs.dx
        a.Assemble(); f.Assemble()
        y.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

        gu = ngs.grad(u)
        F = ngs.BilinearForm(fes)
        F += (ngs.sqrt(gu * gu + 1e-12) * v - 1.0 * v + eps * gu * ngs.grad(v)) * ngs.dx
        ngs.solvers.Newton(F, y, printing=False)
        return y

    def y_normal(self) -> ngs.CoefficientFunction:
        return self._dist_gf

    def wall_distance_cell(self) -> ngs.GridFunction:
        return self._dist_cell

    def _tau_norm_cf(self, U) -> ngs.CoefficientFunction:
        """|tau . n| on the wall (boundary-only CF)."""
        tau = self.mu * (ngs.grad(U) + ngs.grad(U).trans - 2 / 3 * ngs.div(U) * self.IM)
        tau_n = ngs.BoundaryFromVolumeCF(tau) * self.n
        return ngs.sqrt(ngs.InnerProduct(tau_n, tau_n))

    def update_tau_avg(self, U) -> None:
        """Recompute the cross-section-averaged wall shear stress tau_avg(x) and
        store it in self.tau_avg_gf.  Call once per Picard iteration, BEFORE nu_t
        is evaluated.  Bins wall facets by x-centroid -> length-weighted average,
        then extends into the volume as a function of x via VoxelCoefficient."""
        wall_region = self.mesh.Boundaries(self.wall_boundary)
        tau_el = ngs.Integrate(self._tau_norm_cf(U), self.mesh,
                               definedon=wall_region, element_wise=True)
        len_el = ngs.Integrate(ngs.CoefficientFunction(1.0), self.mesh,
                               definedon=wall_region, element_wise=True)

        edges = np.linspace(self.x_min, self.x_max, self.n_xbins + 1)
        num = np.zeros(self.n_xbins)
        den = np.zeros(self.n_xbins)
        for nr, xc in self._wall_els:
            i = int(np.clip(np.searchsorted(edges, xc) - 1, 0, self.n_xbins - 1))
            num[i] += tau_el[nr]
            den[i] += len_el[nr]

        centers = 0.5 * (edges[:-1] + edges[1:])
        ok = den > 0
        tau_x = np.interp(centers, centers[ok], num[ok] / den[ok])
        tau_x = np.maximum(tau_x, 1e-12)

        # VoxelCoefficient expects x as the LAST (fastest) index.
        if self.mesh.dim == 2:
            vals = np.tile(tau_x, (2, 1))
            if self.cylinder_r is None:
                start, end = self._domain_min, self._domain_max
            else:
                start = (self.x_min, -self.cylinder_r)
                end = (self.x_max, self.cylinder_r)
        else:
            vals = np.tile(tau_x, (2, 2, 1))
            if self.cylinder_r is None:
                start, end = self._domain_min, self._domain_max
            else:
                start = (self.x_min, -self.cylinder_r, -self.cylinder_r)
                end = (self.x_max, self.cylinder_r, self.cylinder_r)

        self.tau_avg_gf.Set(ngs.VoxelCoefficient(start, end, vals, linear=True))

    def evaluate_yplus(self, K: ngs.GridFunction, U) -> ngs.CoefficientFunction:
        """y_plus = y_normal * u_t / nu, with u_t = sqrt(tau_avg(x) / rho)."""
        u_t = ngs.sqrt(self.tau_avg_gf / self.rho)
        return self.y_normal() * u_t / self.nu

    def eval_nu_wall_func(self, K: ngs.GridFunction, E, U) -> ngs.CoefficientFunction:
        yplus = self.evaluate_yplus(K, U)
        log = self.nu * (yplus / ((1 / self.kappa) * ngs.log(self.E * yplus)) - 1)
        wall_function = ngs.IfPos((yplus - 11.25) * (200 - yplus), log, 0)
        wf_final = ngs.IfPos(200 - yplus, wall_function, self.C_mu * K ** 2 / E)
        return wf_final

    def eval_nu_t(self, K: ngs.GridFunction, E: ngs.GridFunction, U) -> ngs.CoefficientFunction:
        wall_nu_t = self.eval_nu_wall_func(K, E, U)
        nu_t_bulk = self.C_mu * K ** 2 / E
        nu_t = self.mesh.MaterialCF(
            {self.core_region: nu_t_bulk, self.nearwall_region: wall_nu_t})
        return ngs.IfPos(nu_t, nu_t, 0)  # turbulent viscosity cannot be negative

    def eval_P_k(self, nu_t, U):
        """Standard production term for turbulent kinetic energy."""
        P_k = 2 * (self.nu + nu_t) * ngs.InnerProduct(
            ngs.grad(U), ngs.grad(U) + ngs.grad(U).trans - 2 / 3 * ngs.div(U) * self.IM)
        return ngs.IfPos(P_k, P_k, 0)  # production cannot be negative


class KEpsilonWallFunction:
    """Geometry-independent law-of-the-wall eddy viscosity for ``KEpsilonINS``.

    Numerical contract
    ------------------
    * The near-wall region is *exactly one* layer of volume cells: those owning a
      facet on ``wall_boundary``, found from mesh topology.  No named near-wall or
      core material regions, no cylinder radius, no axis-aligned binning.
    * Wall shear is local and per-cell: the tangential viscous traction averaged
      over that cell's own wall facet(s).  Because the layer is one cell thick
      every marked cell owns a wall facet, so no extension PDE is needed.
    * ``update()`` must be called once per Picard iteration, BEFORE ``eval_nu_t``.
    * Cells outside the mask *always* use bulk k-epsilon viscosity, whatever their
      y+ happens to be.

    Limitations
    -----------
    * y+ is solution dependent and is not validated at construction.  If the first
      cell layer sits outside the log-law's valid band, use :meth:`y_plus_cell` to
      inspect it and adjust the wall-normal mesh size.
    * The y+ distance comes from the Eikonal wall-distance field.  The epsilon
      wall condition separately uses half the local cell size as its first-cell
      distance.
    * Turbulence production and the epsilon wall condition are NOT handled here;
      ``KEpsilonINS`` keeps ownership of both.

    All values are held on ``L2(mesh, order=0)`` and indexed by DOF, never by
    element number -- the linear forms, the mask and the shear field share one DOF
    numbering, so no element-to-DOF mapping is needed internally.
    """

    #: y+ below which the viscous sublayer is assumed (no wall eddy viscosity).
    YPLUS_VISCOUS = 11.25
    #: End of the smooth buffer-to-log-layer transition.
    YPLUS_LOG_MIN = 30.0
    #: Smooth transition from the equilibrium wall law to bulk k-epsilon.

    def __init__(self, mesh: ngs.comp.Mesh, mu: float, rho: float, nu: float,
                 C_mu: float, kappa: float, E: float,
                 wall_boundary: str = "wall",
                 dist_order: int = 2, dist_relax: float = 0.1,
                 tau_floor: float = 1e-12) -> None:
        self.mesh = mesh
        self.mu = mu
        self.rho = rho
        self.nu = nu                      # laminar kinematic viscosity
        self.C_mu = C_mu
        self.kappa = kappa
        self.E = E
        self.wall_boundary = wall_boundary
        self.tau_floor = tau_floor

        self.n = ngs.specialcf.normal(mesh.dim)
        self.h = ngs.specialcf.mesh_size
        self.IM = construct_identity_mat(mesh.dim)

        # Piecewise-constant space shared by the mask, the wall measure, the
        # cell distance and the wall shear.
        self._fes0 = ngs.L2(mesh, order=0)

        # An L2 basis function has no boundary trace, so a plain ds() integral
        # assembles to zero.  skeleton=True evaluates the volume basis function
        # on the facet, which is what puts the wall integral on the owning cell.
        self._wall_ds = ngs.ds(definedon=mesh.Boundaries(wall_boundary),
                               skeleton=True)

        # Per-cell measure (length in 2D, area in 3D) of the cell's wall facets.
        self._wall_measure = self._assemble_wall(ngs.CoefficientFunction(1.0))
        wall_marked = self._wall_measure > 0.0
        if not wall_marked.any():
            raise ValueError(
                f"KEpsilonWallFunction: boundary marker '{wall_boundary}' has no "
                f"boundary elements. Available boundaries: {mesh.GetBoundaries()}.")
        # Keep the true wall-facet owners separate: wall shear and the algebraic
        # boundary integral can be evaluated directly only on these cells.
        self._marked = wall_marked
        self.wall_facet_cell_mask = ngs.GridFunction(self._fes0)
        self.wall_facet_cell_mask.vec.FV().NumPy()[:] = wall_marked.astype(float)

        # Wall-function viscosity mask: wall-facet owners plus one complete
        # face-connected neighbour layer. This suppresses bulk-model production
        # on cells immediately touching the wall layer across an interior facet.
        marked = self._expand_by_one_facet_layer(wall_marked)
        self.mask = ngs.GridFunction(self._fes0)
        self.mask.vec.FV().NumPy()[:] = marked.astype(float)
        self.wall_cell_mask = ngs.GridFunction(self._fes0)
        self.wall_cell_mask.vec.FV().NumPy()[:] = marked.astype(float)

        # Eikonal wall distance used by y+ and diagnostics, projected into the
        # same piecewise-constant space as the mask and wall viscosity.
        self._dist_gf = self._compute_distance_field(dist_order, dist_relax)
        self._dist_cell = ngs.GridFunction(self._fes0)
        self._dist_cell.Set(self._dist_gf)

        # The epsilon wall condition uses the first-cell approximation y_P=h/2,
        # independently of the Eikonal distance used above.
        self._epsilon_dist_cell = ngs.GridFunction(self._fes0)
        self._epsilon_dist_cell.Set(0.5 * self.h)

        # Local wall shear, refreshed by update(). Floor everywhere until then so
        # sqrt() and log() are safe if eval_nu_t runs before the first update.
        self.tau_cell = ngs.GridFunction(self._fes0)
        self.tau_cell.vec.FV().NumPy()[:] = tau_floor

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _assemble_wall(self, integrand: ngs.CoefficientFunction) -> np.ndarray:
        """Integrate ``integrand`` over each cell's wall facets -> DOF-indexed array."""
        test = self._fes0.TestFunction()
        lf = ngs.LinearForm(self._fes0)
        lf += integrand * test * self._wall_ds
        lf.Assemble()
        return np.array(lf.vec).ravel()

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

    def tau_wall_cf(self, U) -> ngs.CoefficientFunction:
        """Magnitude of the *tangential* viscous traction on the wall.

        ``t = sigma.n - n (sigma.n . n)`` removes the normal traction, which the
        legacy ``|sigma.n|`` wrongly included.  The normal's sign cancels.
        """
        sigma = self.mu * (ngs.Grad(U) + ngs.Grad(U).trans
                           - 2 / 3 * ngs.div(U) * self.IM)
        traction = ngs.BoundaryFromVolumeCF(sigma) * self.n
        tangential = traction - self.n * ngs.InnerProduct(traction, self.n)
        # +tau_floor**2 keeps the derivative of sqrt finite at zero traction.
        return ngs.sqrt(ngs.InnerProduct(tangential, tangential) + self.tau_floor ** 2)

    def update(self, U) -> None:
        """Refresh the per-cell wall shear.  Call once per Picard iteration,
        BEFORE ``eval_nu_t``.

        The facet integral is divided by the facet measure for true wall cells.
        Its value is then propagated to the face-connected wall-layer cells;
        cells connected to multiple wall owners receive their arithmetic mean.

        TODO: cells owning more than one wall facet (corners, and any cell with
        two faces on a re-entrant wall) fall out of this as an area-weighted
        aggregate of all their wall facets.  That is a plausible value but it is
        NOT validated -- the tests only cover one wall facet per cell.  Validate
        against a corner case before relying on it.
        """
        shear = self._assemble_wall(self.tau_wall_cf(U))
        tau = np.full_like(shear, self.tau_floor)
        tau[self._marked] = np.maximum(
            shear[self._marked] / self._wall_measure[self._marked], self.tau_floor)
        for element_number, source_numbers in self._wall_layer_sources.items():
            source_values = [
                tau[self._element_dofs[source_number][0]]
                for source_number in source_numbers
            ]
            value = max(float(np.mean(source_values)), self.tau_floor)
            tau[list(self._element_dofs[element_number])] = value
        self.tau_cell.vec.FV().NumPy()[:] = tau

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def near_wall_mask(self) -> ngs.GridFunction:
        """1 on wall cells and their first face-connected neighbours."""
        return self.mask

    def first_cell_mask(self) -> ngs.GridFunction:
        """1 on every cell treated as part of the expanded wall layer."""
        return self.wall_cell_mask

    def wall_facet_mask(self) -> ngs.GridFunction:
        """1 only on cells owning a physical wall facet."""
        return self.wall_facet_cell_mask

    def wall_distance_cell(self) -> ngs.GridFunction:
        return self._dist_cell

    def wall_distance_field(self) -> ngs.GridFunction:
        """Continuous Eikonal wall-distance field used by y+."""
        return self._dist_gf

    def y_plus_cell(self, K) -> ngs.CoefficientFunction:
        """Return cellwise y+ using the high-Re k-epsilon equilibrium relation.

        The friction velocity is ``u_tau = C_mu**0.25 * sqrt(k)``.  It therefore
        depends on the current turbulent kinetic energy rather than the resolved
        molecular traction, which is generally under-resolved on a wall-function
        mesh.
        """
        k_nonnegative = ngs.IfPos(K, K, 0.0)
        u_tau = self.C_mu ** 0.25 * ngs.sqrt(k_nonnegative)
        return self._dist_cell * u_tau / self.nu

    def epsilon_wall_cell(self, K) -> ngs.CoefficientFunction:
        """High-Re equilibrium dissipation in the wall-adjacent cell.

        ``epsilon_P = C_mu**0.75 * k_P**1.5 / (kappa * y_P)``.
        The distance field is cell-averaged and strictly positive in the cells
        that own wall facets.
        """
        k_nonnegative = ngs.IfPos(K, K, 0.0)
        return (self.C_mu ** 0.75 * k_nonnegative ** 1.5
                / (self.kappa * self._epsilon_dist_cell))

    # ------------------------------------------------------------------
    # Eddy viscosity
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_step(value, lower: float, upper: float):
        """C1-continuous transition from zero to one over ``[lower, upper]``."""
        scaled = (value - lower) / (upper - lower)
        clipped = ngs.IfPos(
            scaled, ngs.IfPos(1.0 - scaled, scaled, 1.0), 0.0)
        return clipped ** 2 * (3.0 - 2.0 * clipped)

    def eval_nu_wall(self, K, E) -> ngs.CoefficientFunction:
        """Continuous equilibrium wall-law viscosity in the near-wall layer.

        Constant total shear together with ``du+/dy+ = 1/(kappa*y+)`` gives
        ``nu_t/nu = kappa*y+ - 1``. Smooth transitions avoid the large
        constitutive jumps produced by the legacy hard switches.
        """
        yplus = self.y_plus_cell(K)
        equilibrium = self.nu * (self.kappa * yplus - 1.0)
        equilibrium = ngs.IfPos(equilibrium, equilibrium, 0.0)

        buffer_weight = self._smooth_step(
            yplus, self.YPLUS_VISCOUS, self.YPLUS_LOG_MIN)
        # The topological mask already selects only wall-adjacent cells.  Keep
        # those cells on the wall law for all y+; blending them back to
        # C_mu*k^2/epsilon creates a strong positive feedback when epsilon lags.
        return buffer_weight * equilibrium

    def eval_nu_t(self, K, E, U=None) -> ngs.CoefficientFunction:
        """Blended turbulent viscosity.

        ``U`` is accepted and ignored: the shear it would supply comes from
        :meth:`update`, which must already have run this iteration.
        """
        nu_t_bulk = self.C_mu * K ** 2 / E
        nu_t = self.mask * self.eval_nu_wall(K, E) + (1 - self.mask) * nu_t_bulk
        return ngs.IfPos(nu_t, nu_t, 0)  # turbulent viscosity cannot be negative
