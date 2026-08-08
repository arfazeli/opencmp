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
Bound-preserving scaling limiters for L2 DG scalar GridFunctions.  Provides:

  - p1_vertex_bound : vertex-based scaling limiter (P1 Dunbar basis)
  - bezier_bound     : Bernstein/Bezier maximum-principle-preserving limiter --
                       GUARANTEES the per-element polynomial stays in bounds
                       everywhere (any order).
"""

import ngsolve as ngs
import numpy as np


class Limiter:
    def __init__(self, mesh):
        self.mesh = mesh
        self._bezier_cache = {}  # (id(fes), order) -> built BezierBoundLimiter

    # ── Bezier bound limiter ─────────────────────────────────────────────────
    # Thin accessor over BezierBoundLimiter (defined below). The built limiter
    # precomputes its change-of-basis matrix, so it is cached and reused across
    # calls -- keep the Limiter instance alive to avoid rebuilding.

    def _bezier_limiter(self, fes, order):
        key = (id(fes), order)
        lim = self._bezier_cache.get(key)
        if lim is None:
            lim = BezierBoundLimiter(self.mesh, fes, order)
            self._bezier_cache[key] = lim
        return lim

    def bezier_bound(self, gfu, fes, order, bounds=(0.0, 1.0)):
        '''Bound-preserving scaling limiter via the Bernstein/Bezier convex-hull
        property. Unlike node-sampling limiters, this GUARANTEES the per-element
        polynomial stays in `bounds` everywhere (no between-node leakage).
        Returns number of modified elements.'''
        return self._bezier_limiter(fes, order).apply(gfu, bounds)

    def ref_element_vertices_val(self, gfu: ngs.GridFunction, vertices: np.ndarray,
                                 element_index: int, element_type: str) -> np.ndarray:
        '''
        Evaluates the value of gfu at vertices of a specified element (by gfu_index).
        The coefficients from gfu and Dunbar basis functions (L2) space are used to do the calculation.
        '''
        if element_type == "TRIG":
            [a, b, c] = gfu.vec[element_index: element_index + 3]  # coefficients
            val = (a - b - c) + (3 * b + c) * vertices[:, 0] + (2 * c) * vertices[:, 1]

        if element_type == "TET":
            [a, b, c, d] = gfu.vec[element_index: element_index + 4]  # coefficients
            val = ((a - b - 2 * c - 4 * d) + (4 * b + 2 * c + 4 * d) * vertices[:, 0] + (6 * c + 4 * d) *
                   vertices[:, 1] + 8 * d * vertices[:, 2])
        return val

    def vertices_gfu_val(self, gfu: ngs.GridFunction, element_type: str = "TRIG") -> np.ndarray:
        """
        Evaluates gfu at the vertices of every mesh cell.
        """
        dof_per_element = int(len(gfu.vec) / self.mesh.ne)  # dof per element
        gfu_vertices_val = np.zeros((self.mesh.ne, dof_per_element))

        if element_type == "TRIG":
            vertices = np.array([(0, 0), (1, 0), (0, 1)])
        elif element_type == "TET":
            vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        else:
            raise ValueError("Bound limiter is only implemented for element_type TRIG and TET.")

        for i in range(self.mesh.ne):  # iterate over every mesh element
            gfu_index = i * dof_per_element  # index pointer for coefficients corresponding to the element
            gfu_vertices_val[i, :] = self.ref_element_vertices_val(gfu, vertices, gfu_index, element_type)

        return gfu_vertices_val

    def p1_vertex_bound(self, gfu: ngs.GridFunction, bounds: tuple) -> None:
        '''
        Vertex-based scaling limiter for P1 fields: scales the grid function within each
        element if the max/min value at the mesh cell violates the upper/lower bound.
        (P1 Dunbar reconstruction -- prefer bezier_bound for order >= 2.)
        '''
        (r1, r2) = bounds  # lower and upper bounds
        if self.mesh.dim == 2:
            element_type = "TRIG"
        elif self.mesh.dim == 3:
            element_type = "TET"

        number_of_elements = self.mesh.ne  # number of mesh elements
        dof_per_element = int(len(gfu.vec) / number_of_elements)  # dofs per element
        quad_val_gfu = self.vertices_gfu_val(gfu, element_type)  # gf at the vertices
        quad_min_val, quad_max_val = quad_val_gfu.min(axis=1), quad_val_gfu.max(axis=1)
        theta = np.ones(number_of_elements, dtype=float)  # scaling coefficients

        for i in range(number_of_elements):
            nn = dof_per_element * i  # index of the cell averaged value
            if gfu.vec[nn] < r1:
                gfu.vec[nn] = r1
            elif gfu.vec[nn] > r2:
                gfu.vec[nn] = r2
            if (quad_min_val[i] < r1):
                theta[i] = (gfu.vec[nn] - r1) / (gfu.vec[nn] - quad_min_val[i])
            if (quad_max_val[i] > r2):
                theta2 = (gfu.vec[nn] - r2) / (gfu.vec[nn] - quad_max_val[i])
                theta[i] = min(theta[i], theta2)
            if theta[i] < 1:
                for k in range(1, dof_per_element):
                    gfu.vec[nn + k] = theta[i] * gfu.vec[nn + k]


# ── Reference-element evaluation helpers ─────────────────────────────────────


def _ndof_el(order: int, dim: int) -> int:
    """DOFs per element on the reference simplex."""
    if dim == 2:
        return (order + 1) * (order + 2) // 2
    return (order + 1) * (order + 2) * (order + 3) // 6


def _lagrange_nodes(p: int, dim: int) -> np.ndarray:
    """Uniform Lagrange nodes on the reference simplex for degree p."""
    if p == 0:
        return np.ones((1, dim)) / (dim + 1)   # centroid only
    if dim == 2:
        nodes = [(i/p, j/p)
                 for i in range(p + 1)
                 for j in range(p + 1 - i)]
    else:
        nodes = [(i/p, j/p, k/p)
                 for i in range(p + 1)
                 for j in range(p + 1 - i)
                 for k in range(p + 1 - i - j)]
    return np.array(nodes, dtype=float)


def _build_eval_matrix_ngs(fes: ngs.FESpace, order: int, dim: int,
                            nodes: np.ndarray = None) -> np.ndarray:
    """Build evaluation matrix via NGSolve's FiniteElement.CalcShape."""
    if nodes is None:
        nodes = _lagrange_nodes(order, dim)
    else:
        nodes = np.asarray(nodes, dtype=float)
    ndof_el = _ndof_el(order, dim)
    M  = np.zeros((len(nodes), ndof_el))
    fe = fes.GetFE(ngs.ElementId(ngs.VOL, 0))
    for i, node in enumerate(nodes):
        # CalcShape(x, y, z) evaluates the REFERENCE shape functions directly —
        # geometry-independent, so this is correct for every element regardless
        # of mesh anisotropy/curvature. (Raw-coord overload; pad 2D with z=0.)
        coords = (*node, 0.0) if dim == 2 else tuple(node)
        M[i, :] = np.asarray(fe.CalcShape(*coords))
    return M


def _build_eval_matrix_gf(mesh: ngs.Mesh, fes: ngs.FESpace, order: int, dim: int,
                           nodes: np.ndarray = None) -> np.ndarray:
    """Fallback: build evaluation matrix by probing individual basis vectors.

    Uses a centroid-ward nudge (EPS=1e-10) to keep points strictly inside
    element 0, avoiding ambiguous lookups at shared vertices/faces.
    """
    ndof_el = _ndof_el(order, dim)
    if nodes is None:
        nodes = _lagrange_nodes(order, dim)
    else:
        nodes = np.asarray(nodes, dtype=float)
    M = np.zeros((len(nodes), ndof_el))

    el0   = list(mesh.Elements(ngs.VOL))[0]
    verts = [np.array(list(mesh[v].point)[:dim]) for v in el0.vertices]
    dofs  = fes.GetDofNrs(ngs.ElementId(ngs.VOL, 0))
    gf    = ngs.GridFunction(fes)
    gf_np = gf.vec.FV().NumPy()
    ctr   = np.ones(dim) / (dim + 1)   # reference centroid
    EPS   = 1e-10

    for j in range(ndof_el):
        gf_np[:] = 0.0
        gf_np[int(dofs[j])] = 1.0
        for i, node in enumerate(nodes):
            node_n = node + EPS * (ctr - node)
            bary = np.concatenate(([1.0 - node_n.sum()], node_n))
            phys = sum(b * v for b, v in zip(bary, verts))
            M[i, j] = float(gf(mesh(*phys)))
    return M


def _build_eval_matrix(mesh, fes, order, dim, nodes=None):
    """Try CalcShape first, fall back to GridFunction probing."""
    try:
        return _build_eval_matrix_ngs(fes, order, dim, nodes)
    except Exception:
        return _build_eval_matrix_gf(mesh, fes, order, dim, nodes)


# ── Bound-preserving Bezier/Bernstein limiter ─────────────────────────────────

def _multiindices(p: int, n: int):
    """All length-n nonneg integer tuples summing to p."""
    if n == 1:
        yield (p,)
        return
    for i in range(p + 1):
        for rest in _multiindices(p - i, n - 1):
            yield (i,) + rest


def _bernstein_matrix(nodes: np.ndarray, p: int, dim: int) -> np.ndarray:
    """Degree-p Bernstein basis values at `nodes` on the reference simplex.
    B_alpha(lambda) = (p!/prod alpha_i!) * prod lambda_i^alpha_i."""
    from math import factorial
    idx = list(_multiindices(p, dim + 1))
    M = np.zeros((len(nodes), len(idx)))
    pf = factorial(p)
    for k, node in enumerate(nodes):
        lam = np.concatenate(([1.0 - node.sum()], node))   # barycentric coords
        for j, a in enumerate(idx):
            term = pf
            for d in range(dim + 1):
                term *= lam[d] ** a[d] / factorial(a[d])
            M[k, j] = term
    return M


class BezierBoundLimiter:
    """Maximum-principle-preserving scaling limiter for NGSolve DG on simplices,
    any order. Scales each element's polynomial about its cell mean by a single
    theta computed from the Bernstein/Bezier ordinates. Because the polynomial
    lies within the convex hull of its ordinates, bounding the ordinates bounds
    the polynomial EVERYWHERE — not just at sample nodes (the failure mode of
    node-sampling limiters on high-order, high-curvature near-wall cells)."""

    def __init__(self, mesh: ngs.Mesh, fes: ngs.FESpace, order: int = 1):
        self.mesh    = mesh
        self.order   = order
        self.dim     = mesh.dim
        self.ndof_el = _ndof_el(order, self.dim)
        # Change-of-basis: L2 element dofs -> Bezier ordinates. Built once on the
        # reference element. b = T @ c, where v = Lmat@c = Bmat@b at unisolvent
        # degree-p Lagrange nodes, so T = Bmat^{-1} @ Lmat.
        ref_nodes  = _lagrange_nodes(order, self.dim)            # ndof_el points
        Lmat       = _build_eval_matrix(mesh, fes, order, self.dim, nodes=ref_nodes)
        Bmat       = _bernstein_matrix(ref_nodes, order, self.dim)
        self._to_bezier = np.linalg.solve(Bmat, Lmat)
        self.dof_starts = np.array([
            fes.GetDofNrs(ngs.ElementId(ngs.VOL, i))[0]
            for i in range(mesh.ne)
        ], dtype=np.intp)

    def apply(self, gfu: ngs.GridFunction, bounds: tuple = (0.0, 1.0)) -> int:
        r1, r2 = bounds
        nd     = self.ndof_el
        vec    = gfu.vec.FV().NumPy()
        n_lim  = 0

        for base in self.dof_starts:
            c   = vec[base : base + nd]
            ubar = c[0]                      # cell mean (NGSolve L2: phi0 == 1)

            # bound the average first
            if ubar < r1 or ubar > r2:
                ubar = float(np.clip(ubar, r1, r2))
                vec[base] = ubar
                vec[base + 1 : base + nd] = 0.0
                n_lim += 1
                continue

            b = self._to_bezier @ c          # Bezier ordinates: poly in [b.min, b.max]
            bmin, bmax = float(b.min()), float(b.max())

            theta = 1.0
            if bmax > r2 + 1e-14:
                theta = min(theta, (r2 - ubar) / (bmax - ubar))
            if bmin < r1 - 1e-14:
                theta = min(theta, (r1 - ubar) / (bmin - ubar))
            theta = max(0.0, min(1.0, theta))

            if theta < 1.0 - 1e-14:
                vec[base + 1 : base + nd] *= theta
                n_lim += 1

        return n_lim
