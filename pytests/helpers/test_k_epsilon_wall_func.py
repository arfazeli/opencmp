"""Tests for the geometry-independent k-epsilon wall function.

The point of ``KEpsilonWallFunction`` is that it needs no cylinder radius and no
hand-labelled near-wall/core materials, so most of these tests check that the
near-wall layer, the wall distance and the wall law come out right on meshes
that carry none of that information.
"""

import numpy as np
import ngsolve as ngs
import pytest
from netgen.csg import unit_cube
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

from opencmp.helpers.wall_func import KEpsilonWallFunction

SQUARE = 'pytests/mesh_files/unit_square_coarse.vol'
CHANNEL = 'pytests/mesh_files/channel_3bcs.vol'

NU = 1.5
E_LOG = 9.8


def build(mesh, wall='bottom', **kwargs):
    return KEpsilonWallFunction(mesh, nu=NU, C_mu=0.09, kappa=0.41, E_log=E_LOG,
                                wall_boundary=wall, **kwargs)


def marked_element_numbers(wf):
    """Element numbers of the marked cells, via the space's element->DOF map.

    Deliberately does not assume DOF number == element number.
    """
    values = wf.mask.vec.FV().NumPy()
    return {el.nr for el in wf._fes0.Elements(ngs.VOL)
            if any(values[dof] > 0.5 for dof in el.dofs)}


def elements_owning_a_facet_on(mesh, wall):
    """Ground truth: an element owns a wall facet when at least ``dim`` of its
    vertices lie on that boundary (``dim - 1`` vertices is a vertex/edge touch)."""
    wall_vertices = set()
    for sel in mesh.Elements(ngs.BND):
        if sel.mat == wall:
            wall_vertices.update(v.nr for v in sel.vertices)
    return {el.nr for el in mesh.Elements(ngs.VOL)
            if sum(1 for v in el.vertices if v.nr in wall_vertices) >= mesh.dim}


def add_one_face_connected_layer(mesh, element_numbers):
    """Ground truth for one topological dilation through volume-cell facets."""
    elements = list(mesh.Elements(ngs.VOL))
    facet_to_elements = {}
    element_by_number = {element.nr: element for element in elements}
    for element in elements:
        for facet in element.facets:
            facet_to_elements.setdefault(facet.nr, set()).add(element.nr)

    expanded = set(element_numbers)
    for element_number in element_numbers:
        for facet in element_by_number[element_number].facets:
            expanded.update(facet_to_elements[facet.nr])
    return expanded


# ----------------------------------------------------------------------
# Topology: the near-wall mask
# ----------------------------------------------------------------------

@pytest.mark.parametrize('meshfile, wall', [(SQUARE, 'bottom'), (CHANNEL, 'wall')])
def test_mask_includes_wall_cells_and_one_face_connected_layer(meshfile, wall):
    mesh = ngs.Mesh(meshfile)
    wf = build(mesh, wall)
    wall_cells = elements_owning_a_facet_on(mesh, wall)
    expected = add_one_face_connected_layer(mesh, wall_cells)
    assert marked_element_numbers(wf) == expected


def test_near_wall_mask_matches_the_full_expanded_wall_layer():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    values = wf.near_wall_mask().vec.FV().NumPy()
    actual = {element.nr for element in wf._fes0.Elements(ngs.VOL)
              if any(values[dof] > 0.5 for dof in element.dofs)}
    wall_cells = elements_owning_a_facet_on(mesh, 'wall')
    assert actual == add_one_face_connected_layer(mesh, wall_cells)


def test_wall_facet_mask_retains_the_true_boundary_owners():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    values = wf.wall_facet_mask().vec.FV().NumPy()
    actual = {element.nr for element in wf._fes0.Elements(ngs.VOL)
              if any(values[dof] > 0.5 for dof in element.dofs)}
    assert actual == elements_owning_a_facet_on(mesh, 'wall')


def test_mask_is_not_empty_and_is_a_strict_subset():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    marked = marked_element_numbers(wf)
    assert 0 < len(marked) < mesh.ne


def test_wall_measure_sums_to_the_exact_boundary_measure():
    """Every wall facet is attributed to exactly one cell -- no double counting,
    none dropped."""
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    exact = ngs.Integrate(ngs.CoefficientFunction(1.0), mesh,
                          definedon=mesh.Boundaries('wall'))
    assert wf._wall_measure.sum() == pytest.approx(exact, rel=1e-12)


def test_cell_touching_the_wall_only_at_a_vertex_is_not_marked():
    """A vertex-only touch is excluded unless it shares a facet with layer one."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    marked = marked_element_numbers(wf)

    bottom_vertices = set()
    for sel in mesh.Elements(ngs.BND):
        if sel.mat == 'bottom':
            bottom_vertices.update(v.nr for v in sel.vertices)

    vertex_only = {el.nr for el in mesh.Elements(ngs.VOL)
                   if sum(1 for v in el.vertices if v.nr in bottom_vertices) == 1}
    assert vertex_only, 'mesh exercises no vertex-only touch; test is vacuous'
    wall_cells = elements_owning_a_facet_on(mesh, 'bottom')
    expected = add_one_face_connected_layer(mesh, wall_cells)
    assert vertex_only & marked == vertex_only & expected


def test_missing_wall_marker_raises_a_clear_error():
    mesh = ngs.Mesh(SQUARE)
    with pytest.raises(ValueError, match='no boundary elements'):
        build(mesh, 'not_a_boundary')


# ----------------------------------------------------------------------
# Wall distance
# ----------------------------------------------------------------------

def test_wall_distance_is_zero_on_the_wall_and_grows_inward():
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    dist = wf.wall_distance_field()

    assert abs(dist(mesh(0.5, 0.0))) < 1e-8
    samples = [dist(mesh(0.5, y)) for y in (0.1, 0.3, 0.6, 0.9)]
    assert all(b > a for a, b in zip(samples, samples[1:]))
    assert samples[0] == pytest.approx(0.1, abs=0.05)


def _epsilon_wall_values(wf, k):
    epsilon = ngs.GridFunction(wf._fes0)
    epsilon.Set(wf.epsilon_wall_cell(ngs.CoefficientFunction(k)))
    return epsilon.vec.FV().NumPy()


def _k_for_yplus(wf, yplus):
    """k giving the requested y+ in the *first* cell, from y+ = y*Cmu^0.25*sqrt(k)/nu."""
    y = float(wf.wall_distance_cell().vec.FV().NumPy().min())
    return (yplus * NU / (0.09 ** 0.25 * y)) ** 2


def test_epsilon_wall_uses_high_re_equilibrium_relation_in_the_log_layer():
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    k = _k_for_yplus(wf, 5.0 * wf.YPLUS_VISCOUS)
    distance = wf.wall_distance_cell().vec.FV().NumPy()
    expected = 0.09 ** 0.75 * k ** 1.5 / (0.41 * distance)

    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert _epsilon_wall_values(wf, k)[owners] == pytest.approx(
        expected[owners], rel=1e-9)


def test_epsilon_wall_switches_to_viscous_dissipation_below_yplus_lam():
    """Below y+_lam production is zero, so log-layer dissipation would empty the cell."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    k = _k_for_yplus(wf, 0.1 * wf.YPLUS_VISCOUS)
    distance = wf.wall_distance_cell().vec.FV().NumPy()
    expected = 2.0 * NU * k / distance ** 2

    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert _epsilon_wall_values(wf, k)[owners] == pytest.approx(
        expected[owners], rel=1e-9)


def wall_law_nu_t(yplus):
    """Log-law wall viscosity: nu*(y+/u+ - 1) with u+ = ln(E*y+)/kappa."""
    return NU * (yplus / (np.log(E_LOG * yplus) / 0.41) - 1.0)


def test_marked_cells_are_the_closest_cells_to_the_wall():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    dist = wf.wall_distance_cell().vec.FV().NumPy()
    marked = wf.mask.vec.FV().NumPy() > 0.5
    assert dist[marked].max() < dist[~marked].max()


# ----------------------------------------------------------------------
# Viscosity selection
# ----------------------------------------------------------------------

def force_yplus(wf, target):
    """Return cellwise k values that give target y+ on every marked cell."""
    dist = wf.wall_distance_cell().vec.FV().NumPy()
    marked = wf.mask.vec.FV().NumPy() > 0.5
    k = ngs.GridFunction(wf._fes0)
    values = k.vec.FV().NumPy()
    values[:] = 1.0
    values[marked] = (target * NU / (wf.C_mu ** 0.25 * dist[marked])) ** 2
    return marked, k


def sample_nu_t(wf, mesh, k, eps):
    """nu_t as a cellwise array, via an L2(0) projection."""
    K = ngs.CoefficientFunction(k)
    E = ngs.CoefficientFunction(eps)
    out = ngs.GridFunction(wf._fes0)
    out.Set(wf.eval_nu_t(K, E))
    return out.vec.FV().NumPy()


def test_compiled_viscosity_matches_symbolic_viscosity():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    _, k = force_yplus(wf, 50.0)
    epsilon = ngs.CoefficientFunction(1.0)
    symbolic = wf.eval_nu_t(k, epsilon)
    compiled = symbolic.Compile()
    symbolic_values = ngs.GridFunction(wf._fes0)
    compiled_values = ngs.GridFunction(wf._fes0)
    symbolic_values.Set(symbolic)
    compiled_values.Set(compiled)

    assert compiled_values.vec.FV().NumPy() == pytest.approx(
        symbolic_values.vec.FV().NumPy(), rel=1e-12, abs=1e-12)


@pytest.fixture
def channel_wf():
    mesh = ngs.Mesh(CHANNEL)
    return mesh, build(mesh, 'wall')


def test_low_yplus_selects_zero_wall_viscosity(channel_wf):
    mesh, wf = channel_wf
    marked, k = force_yplus(wf, 5.0)                  # below 11.25
    nu_t = sample_nu_t(wf, mesh, k=k, eps=1.0)
    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert nu_t[owners] == pytest.approx(0.0, abs=1e-12)


def test_log_range_yplus_selects_the_log_law(channel_wf):
    mesh, wf = channel_wf
    yplus = 50.0
    marked, k = force_yplus(wf, yplus)
    nu_t = sample_nu_t(wf, mesh, k=k, eps=1.0)
    expected = wall_law_nu_t(yplus)
    assert expected > 0
    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert nu_t[owners] == pytest.approx(expected, rel=1e-6)


def test_wall_viscosity_varies_across_the_wall_cell(channel_wf):
    """eval_nu_wall takes y+ from the continuous Eikonal field, so the coefficient
    varies inside the element instead of being one number per cell.  The target y+
    keeps the whole cell inside the log-law band, so the variation measured here is
    the log law's own, not the switch to bulk at YPLUS_LOG_MAX."""
    mesh, wf = channel_wf
    _, k = force_yplus(wf, 60.0)
    nu_t = wf.eval_nu_wall(ngs.CoefficientFunction(k), ngs.CoefficientFunction(1.0))

    cellwise_mean = ngs.GridFunction(wf._fes0)
    cellwise_mean.Set(nu_t)
    spread = ngs.Integrate((nu_t - cellwise_mean) ** 2, mesh)
    magnitude = ngs.Integrate(cellwise_mean ** 2, mesh)

    assert np.sqrt(spread / magnitude) > 0.05


def test_wall_viscosity_is_continuous_at_the_sublayer_threshold(channel_wf):
    """The log law is <= 0 at YPLUS_VISCOUS and clamped to zero, so the sublayer
    branch joins it continuously."""
    mesh, wf = channel_wf
    threshold = wf.YPLUS_VISCOUS
    samples = []
    for yplus in (threshold - 1e-5, threshold + 1e-5):
        marked, k = force_yplus(wf, yplus)
        values = sample_nu_t(wf, mesh, k=k, eps=wf.epsilon_wall_cell(k))
        samples.append(values[marked])

    scale = np.maximum(np.maximum(np.abs(samples[0]), np.abs(samples[1])), 1.0)
    assert np.max(np.abs(samples[1] - samples[0]) / scale) < 1e-4


def test_yplus_above_the_log_law_limit_falls_back_to_bulk(channel_wf):
    """Past YPLUS_LOG_MAX the log law no longer applies, so even wall cells take
    the bulk C_mu*k^2/epsilon value."""
    mesh, wf = channel_wf
    marked, k = force_yplus(wf, 5.0 * wf.YPLUS_LOG_MAX)
    eps = 2.5
    nu_t = sample_nu_t(wf, mesh, k=k, eps=eps)

    kvals = k.vec.FV().NumPy()
    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert nu_t[owners] == pytest.approx(0.09 * kvals[owners] ** 2 / eps, rel=1e-6)


def test_wall_law_holds_across_the_whole_log_layer(channel_wf):
    mesh, wf = channel_wf
    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    for yplus in (20.0, 60.0, 150.0, 199.0):
        _, k = force_yplus(wf, yplus)
        nu_t = sample_nu_t(wf, mesh, k=k, eps=2.5)
        assert nu_t[owners] == pytest.approx(wall_law_nu_t(yplus), rel=1e-6), yplus


def test_unmarked_cells_use_bulk_even_when_their_yplus_is_low(channel_wf):
    """Cells outside the expanded wall layer use bulk k-epsilon regardless of y+."""
    mesh, wf = channel_wf
    marked, k = force_yplus(wf, 5.0)                  # wall term would be 0
    k.vec.FV().NumPy()[~marked] = 1e-12               # also low y+ outside mask
    eps = 2.5
    nu_t = sample_nu_t(wf, mesh, k=k, eps=eps)

    kvals = k.vec.FV().NumPy()
    bulk = 0.09 * kvals ** 2 / eps
    assert nu_t[~marked] == pytest.approx(bulk[~marked], rel=1e-6)
    owners = wf.wall_facet_mask().vec.FV().NumPy() > 0.5
    assert nu_t[owners] == pytest.approx(0.0, abs=1e-12)

    yplus = ngs.GridFunction(wf._fes0)
    yplus.Set(wf.y_plus_cell(k))
    assert yplus.vec.FV().NumPy()[~marked].min() < 11.25, \
        'no low-y+ unmarked cell; test is vacuous'


def test_bulk_viscosity_matches_the_plain_k_epsilon_formula(channel_wf):
    mesh, wf = channel_wf
    k, eps = 0.8, 1.3
    nu_t = sample_nu_t(wf, mesh, k=k, eps=eps)
    marked = wf.mask.vec.FV().NumPy() > 0.5
    assert nu_t[~marked] == pytest.approx(0.09 * k ** 2 / eps, rel=1e-6)


def test_wall_layer_neighbours_inherit_wall_owner_friction_velocity(channel_wf):
    _, wf = channel_wf
    k = ngs.GridFunction(wf._fes0)
    values = k.vec.FV().NumPy()
    values[:] = 1e6  # Must not determine u_tau in non-owner wall-layer cells.
    values[wf._marked] = np.linspace(0.2, 0.8, wf._marked.sum())

    wf.update(k)
    u_tau = wf.u_tau_cell.vec.FV().NumPy()
    for element_number, source_numbers in wf._wall_layer_sources.items():
        expected = np.mean([
            u_tau[wf._element_dofs[source_number][0]]
            for source_number in source_numbers
        ])
        assert u_tau[list(wf._element_dofs[element_number])] == pytest.approx(
            expected, rel=1e-12)


def test_negative_viscosity_is_clamped_to_zero(channel_wf):
    mesh, wf = channel_wf
    nu_t = sample_nu_t(wf, mesh, k=1.0, eps=-1.0)     # negative bulk
    assert nu_t.min() >= 0.0


# ----------------------------------------------------------------------
# Geometry independence
# ----------------------------------------------------------------------

def test_runs_on_a_mesh_with_no_named_regions_or_cylinder_radius():
    """channel_3bcs has a single 'default' material and no near-wall/core
    labelling; the legacy class could not have been built on it."""
    mesh = ngs.Mesh(CHANNEL)
    assert set(mesh.GetMaterials()) == {'default'}
    wf = build(mesh, 'wall')
    wf.update(ngs.CoefficientFunction(1.0))
    assert wf.u_tau_cell.vec.FV().NumPy().max() > 0


# ----------------------------------------------------------------------
# Element type: simplices only
# ----------------------------------------------------------------------

@pytest.fixture
def cube_wf():
    """Unit cube of tets, wall on the z = 0 face."""
    mesh = ngs.Mesh(unit_cube.GenerateMesh(maxh=0.25))
    return mesh, build(mesh, 'bottom')


def test_tet_mesh_attributes_every_wall_face_to_exactly_one_cell(cube_wf):
    mesh, wf = cube_wf
    exact = ngs.Integrate(ngs.CoefficientFunction(1.0), mesh,
                          definedon=mesh.Boundaries('bottom'))
    assert wf._wall_measure.sum() == pytest.approx(exact, rel=1e-12)


def test_tet_mask_is_wall_face_owners_plus_one_face_connected_layer(cube_wf):
    mesh, wf = cube_wf
    wall_cells = elements_owning_a_facet_on(mesh, 'bottom')
    assert wall_cells, 'no cell owns a wall face; test is vacuous'
    assert marked_element_numbers(wf) == add_one_face_connected_layer(mesh, wall_cells)


def test_tet_marked_cells_are_the_closest_cells_to_the_wall(cube_wf):
    _, wf = cube_wf
    dist = wf.wall_distance_cell().vec.FV().NumPy()
    marked = wf.mask.vec.FV().NumPy() > 0.5
    assert dist[marked].max() < dist[~marked].max()


def test_tet_wall_distance_is_zero_on_the_wall_and_grows_inward(cube_wf):
    mesh, wf = cube_wf
    dist = wf.wall_distance_field()
    assert abs(dist(mesh(0.5, 0.5, 0.0))) < 1e-8
    samples = [dist(mesh(0.5, 0.5, z)) for z in (0.1, 0.2, 0.3)]
    assert all(b > a for a, b in zip(samples, samples[1:]))
    assert samples[0] == pytest.approx(0.1, abs=0.05)


def test_tet_friction_velocity_is_the_equilibrium_value_on_owners(cube_wf):
    _, wf = cube_wf
    k = 0.5
    wf.update(ngs.CoefficientFunction(k))
    u_tau = wf.u_tau_cell.vec.FV().NumPy()
    assert u_tau[wf._marked] == pytest.approx(0.09 ** 0.25 * np.sqrt(k), rel=1e-9)
    assert u_tau[wf.mask.vec.FV().NumPy() < 0.5].max() == 0.0


def test_tet_cells_outside_the_mask_use_bulk_viscosity(cube_wf):
    mesh, wf = cube_wf
    k, eps = 0.5, 1.0
    wf.update(ngs.CoefficientFunction(k))
    nu_t = ngs.GridFunction(wf._fes0)
    nu_t.Set(wf.eval_nu_t(ngs.CoefficientFunction(k), ngs.CoefficientFunction(eps)))
    outside = wf.mask.vec.FV().NumPy() < 0.5
    assert nu_t.vec.FV().NumPy()[outside] == pytest.approx(0.09 * k ** 2 / eps, rel=1e-9)


@pytest.mark.parametrize('mesh_factory, kind', [
    (lambda: MakeStructured2DMesh(quads=True, nx=4, ny=4), 'QUAD'),
    (lambda: MakeStructured3DMesh(hexes=True, nx=3, ny=3, nz=3), 'HEX'),
])
def test_non_simplicial_meshes_are_refused(mesh_factory, kind):
    """Wall functions are not implemented for quads or hexes."""
    mesh = mesh_factory()
    with pytest.raises(NotImplementedError, match=kind):
        build(mesh, 'bottom')
