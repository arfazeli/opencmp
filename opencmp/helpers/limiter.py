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
Slope / bound-preserving limiters for L2 DG scalar GridFunctions.  Provides:

  - bound_limiter_Joshaghani : vertex-based scaling limiter (P1 Dunbar basis)
  - barth_jespersen          : Barth-Jespersen / Venkatakrishnan (any order)
  - kuzmin                   : Kuzmin (2010) vertex-based (any order)
  - bezier_bound             : Bernstein/Bezier maximum-principle-preserving
                               limiter -- GUARANTEES the per-element polynomial
                               stays in bounds everywhere (any order).
"""

import ngsolve as ngs
import numpy as np


class Limiter:
    def __init__(self, mesh):
        self.mesh = mesh
        self._slope_cache = {}  # (id(fes), order, kind) -> built slope limiter

    # ── Vertex-based / Barth-Jespersen slope limiters ───────────────────────────
    # Thin accessors over BJLimiter2D / KuzminLimiter2D (defined below). The built
    # limiter precomputes eval matrices + adjacency tables, so it is cached and
    # reused across calls — keep the Limiter instance alive to avoid rebuilding.

    def _slope_limiter(self, kind, fes, order):
        key = (id(fes), order, kind)
        lim = self._slope_cache.get(key)
        if lim is None:
            cls = {"bj": BJLimiter2D, "kuzmin": KuzminLimiter2D,
                   "bezier": BezierBoundLimiter}[kind]
            lim = cls(self.mesh, fes, order)
            self._slope_cache[key] = lim
        return lim

    def barth_jespersen(self, gfu, fes, order, bounds=(0.0, 1.0), **kwargs):
        '''Barth-Jespersen / Venkatakrishnan slope limiter (TRIG/TET, any order).
        Returns number of modified elements. See BJLimiter2D.apply for kwargs
        (use_indicator, indicator_threshold, venkat_eps).'''
        return self._slope_limiter("bj", fes, order).apply(gfu, bounds, **kwargs)

    def bezier_bound(self, gfu, fes, order, bounds=(0.0, 1.0)):
        '''Bound-preserving scaling limiter via the Bernstein/Bezier convex-hull
        property. Unlike node-sampling limiters, this GUARANTEES the per-element
        polynomial stays in `bounds` everywhere (no between-node leakage).
        Returns number of modified elements.'''
        return self._slope_limiter("bezier", fes, order).apply(gfu, bounds)

    def kuzmin(self, gfu, fes, order, bounds=(0.0, 1.0)):
        '''Kuzmin (2010) vertex-based slope limiter (TRIG/TET, any order).
        Less diffusive than BJ. Returns number of modified elements.'''
        return self._slope_limiter("kuzmin", fes, order).apply(gfu, bounds)

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

    def bound_limiter_Joshaghani(self, gfu: ngs.GridFunction, bounds: tuple) -> None:
        '''
        Scales the grid function within each element if the max/min value at the mesh cell violates the
        upper/lower bound.  (P1 Dunbar reconstruction — prefer bezier_bound for order >= 2.)
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


# ── Vertex-based slope limiters ──────────────────────────────────────────────


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


# Reference simplex vertices keyed by dimension
_VERTEX_NODES = {
    2: np.array([[0., 0.], [1., 0.], [0., 1.]]),
    3: np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
}


def _vertex_clustered_nodes(dim: int,
                            fracs: tuple = (0.05, 0.12, 0.25)) -> np.ndarray:
    """Extra check nodes packed near each reference-simplex vertex, where a
    degree>=2 polynomial overshoots most. For every vertex, places points a
    small fraction `t` of the way toward each other vertex and the centroid."""
    V = _VERTEX_NODES[dim]
    targets = list(V) + [V.mean(axis=0)]   # other vertices + centroid
    pts = [v + t * (tgt - v)
           for v in V for tgt in targets for t in fracs
           if not np.allclose(v, tgt)]
    return np.array(pts, dtype=float)


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


# ── Neighbour tables ──────────────────────────────────────────────────────────

def build_neighbor_table(mesh: ngs.Mesh) -> list:
    """Face-adjacency by iteration index (0..ne-1).

    Two simplices share a face when they share dim vertices:
      dim=2 (triangles): shared edge  = 2 shared vertices
      dim=3 (tets):      shared face  = 3 shared vertices
    """
    dim     = mesh.dim
    el_list = list(mesh.Elements(ngs.VOL))
    v2e: dict = {}
    for idx, el in enumerate(el_list):
        for v in el.vertices:
            v2e.setdefault(v.nr, []).append(idx)

    neighbors = [[] for _ in range(mesh.ne)]
    for idx, el in enumerate(el_list):
        shared: dict = {}
        for v in el.vertices:
            for nb_idx in v2e[v.nr]:
                if nb_idx != idx:
                    shared[nb_idx] = shared.get(nb_idx, 0) + 1
        for nb_idx, cnt in shared.items():
            if cnt == dim:
                neighbors[idx].append(nb_idx)
    return neighbors


def build_vertex_star_table(mesh: ngs.Mesh) -> list:
    """For each element, the list of elements sharing each of its vertices
    (used by KuzminLimiter2D for per-vertex bounds). Works for any dimension."""
    el_list = list(mesh.Elements(ngs.VOL))
    v2e: dict = {}
    for idx, el in enumerate(el_list):
        for v in el.vertices:
            v2e.setdefault(v.nr, []).append(idx)
    return [[v2e[v.nr] for v in el.vertices] for el in el_list]


# ── BJ Limiter ────────────────────────────────────────────────────────────────

class BJLimiter2D:
    """Barth-Jespersen slope limiter for NGSolve DG.
    Works on triangular (2D) and tetrahedral (3D) meshes, any polynomial order.
    """

    def __init__(self, mesh: ngs.Mesh, fes: ngs.FESpace, order: int = 1):
        self.mesh      = mesh
        self.order     = order
        self.dim       = mesh.dim
        self.ndof_el   = _ndof_el(order, self.dim)
        # Check nodes: uniform 2*order oversample + extra nodes clustered near
        # vertices (where degree>=2 polynomials overshoot most). Dedup so the
        # eval matrix stays small.
        _check_nodes   = np.vstack([
            _VERTEX_NODES[self.dim],
            _lagrange_nodes(max(2 * order, 1), self.dim),
            _vertex_clustered_nodes(self.dim),
        ])
        _check_nodes   = np.unique(np.round(_check_nodes, 12), axis=0)
        self._eval_mat = _build_eval_matrix(mesh, fes, order, self.dim,
                                            nodes=_check_nodes)
        self.neighbors = build_neighbor_table(mesh)
        self.dof_starts = np.array([
            fes.GetDofNrs(ngs.ElementId(ngs.VOL, i))[0]
            for i in range(mesh.ne)
        ], dtype=np.intp)

    def _cell_averages(self, gfu: ngs.GridFunction) -> np.ndarray:
        return gfu.vec.FV().NumPy()[self.dof_starts].copy()

    def troubled_cells(self, u0: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Boolean mask: element i is troubled if the cell-average jump across
        any face exceeds threshold × (global variation)."""
        var = float(u0.max() - u0.min())
        if var < 1e-14:
            return np.zeros(len(u0), dtype=bool)
        thr      = threshold * var
        troubled = np.zeros(len(u0), dtype=bool)
        for i, nbs in enumerate(self.neighbors):
            if nbs and float(np.max(np.abs(u0[i] - u0[nbs]))) > thr:
                troubled[i] = True
        return troubled

    @staticmethod
    def _theta(uv: float, ubar: float, u_bound: float, eps2: float) -> float:
        """Per-node θ: standard BJ (eps2=0) or Venkatakrishnan smooth limiter."""
        b = uv - ubar
        if abs(b) < 1e-14:
            return 1.0
        a = u_bound - ubar
        if a * b <= 0.0:
            return 1.0
        if eps2 == 0.0:
            return min(1.0, a / b)
        return (a*a + 2.0*a*b + eps2) / (a*a + a*b + 2.0*b*b + eps2)

    def apply(self, gfu: ngs.GridFunction,
              bounds: tuple = (0.0, 1.0),
              use_indicator: bool = True,
              indicator_threshold: float = 0.1,
              venkat_eps: float = 0.0) -> int:
        """Apply BJ / Venkatakrishnan limiter in-place. Returns number of modified elements.

        venkat_eps: 0.0 = standard BJ (hard), 0.3 = recommended smooth, 1.0 = very smooth.
        """
        r1, r2   = bounds
        nd       = self.ndof_el
        ne       = self.mesh.ne
        u0       = self._cell_averages(gfu)
        vec      = gfu.vec.FV().NumPy()
        troubled = (self.troubled_cells(u0, indicator_threshold)
                    if use_indicator else np.ones(ne, dtype=bool))
        var      = float(u0.max() - u0.min())
        eps2     = (venkat_eps * var) ** 2
        n_lim    = 0

        for i in range(ne):
            base = self.dof_starts[i]
            ū    = u0[i]

            if ū < r1 or ū > r2:
                vec[base] = float(np.clip(ū, r1, r2))
                vec[base + 1 : base + nd] = 0.0
                n_lim += 1
                continue

            dofs    = vec[base : base + nd].copy()
            u_nodes = self._eval_mat @ dofs

            theta = 1.0
            if troubled[i]:
                nbs = self.neighbors[i]
                if nbs:
                    nb_u0 = u0[nbs]
                    u_max = min(r2, max(ū, float(nb_u0.max())))
                    u_min = max(r1, min(ū, float(nb_u0.min())))
                else:
                    u_max, u_min = float(r2), float(r1)
                for uv in u_nodes:
                    if uv > ū + 1e-12:
                        theta = min(theta, self._theta(uv, ū, u_max, eps2))
                    elif uv < ū - 1e-12:
                        theta = min(theta, self._theta(uv, ū, u_min, eps2))

            # Hard global bound — applied to all cells
            theta_gb = 1.0
            for uv in u_nodes:
                if uv > r2 + 1e-12 and uv > ū + 1e-12:
                    theta_gb = min(theta_gb, (r2 - ū) / (uv - ū))
                elif uv < r1 - 1e-12 and uv < ū - 1e-12:
                    theta_gb = min(theta_gb, (r1 - ū) / (uv - ū))
            theta_gb = max(0.0, min(1.0, theta_gb))
            theta    = min(max(0.0, min(1.0, theta)), theta_gb)

            if theta < 1.0 - 1e-14:
                vec[base + 1 : base + nd] *= theta
                n_lim += 1

        return n_lim


# ── Kuzmin Vertex-Based Limiter ───────────────────────────────────────────────

class KuzminLimiter2D:
    """Kuzmin (2010) vertex-based slope limiter for NGSolve DG.
    Works on triangular (2D) and tetrahedral (3D) meshes, any polynomial order.

    Less diffusive than BJ: bounds come from the wider vertex star and only
    the simplex vertices (3 in 2D, 4 in 3D) are checked instead of all Lagrange nodes.
    """

    def __init__(self, mesh: ngs.Mesh, fes: ngs.FESpace, order: int = 1):
        self.mesh      = mesh
        self.order     = order
        self.dim       = mesh.dim
        self.ndof_el   = _ndof_el(order, self.dim)
        self._vert_eval = _build_eval_matrix(mesh, fes, order, self.dim,
                                             nodes=_VERTEX_NODES[self.dim])
        self.vertex_star = build_vertex_star_table(mesh)
        self.dof_starts  = np.array([
            fes.GetDofNrs(ngs.ElementId(ngs.VOL, i))[0]
            for i in range(mesh.ne)
        ], dtype=np.intp)

    def _cell_averages(self, gfu: ngs.GridFunction) -> np.ndarray:
        return gfu.vec.FV().NumPy()[self.dof_starts].copy()

    def apply(self, gfu: ngs.GridFunction,
              bounds: tuple = (0.0, 1.0)) -> int:
        """Apply Kuzmin vertex-based limiter in-place. Returns number of modified elements."""
        r1, r2 = bounds
        nd     = self.ndof_el
        ne     = self.mesh.ne
        u0     = self._cell_averages(gfu)
        vec    = gfu.vec.FV().NumPy()
        n_lim  = 0

        for i in range(ne):
            base = self.dof_starts[i]
            ū    = u0[i]

            if ū < r1 or ū > r2:
                vec[base] = float(np.clip(ū, r1, r2))
                vec[base + 1 : base + nd] = 0.0
                n_lim += 1
                continue

            dofs    = vec[base : base + nd].copy()
            u_verts = self._vert_eval @ dofs   # (3,) in 2D, (4,) in 3D

            theta = 1.0
            for j, uv in enumerate(u_verts):
                star    = self.vertex_star[i][j]
                u_max_v = min(r2, float(u0[star].max()))
                u_min_v = max(r1, float(u0[star].min()))
                if uv > ū + 1e-12:
                    theta = min(theta, (u_max_v - ū) / (uv - ū))
                elif uv < ū - 1e-12:
                    theta = min(theta, (u_min_v - ū) / (uv - ū))

            theta = max(0.0, min(1.0, theta))
            if theta < 1.0 - 1e-14:
                vec[base + 1 : base + nd] *= theta
                n_lim += 1

        return n_lim


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
    BJ/Kuzmin on high-order, high-curvature near-wall cells)."""

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
