"""Tests for the geometry-independent k-epsilon wall function.

The point of ``KEpsilonWallFunction`` is that it needs no cylinder radius and no
hand-labelled near-wall/core materials, so most of these tests check that the
near-wall layer, the wall distance and the wall shear come out right on meshes
that carry none of that information.
"""

import numpy as np
import ngsolve as ngs
import pytest

from opencmp.helpers.wall_func import KEpsilonWallFunction

SQUARE = 'pytests/mesh_files/unit_square_coarse.vol'
CHANNEL = 'pytests/mesh_files/channel_3bcs.vol'

MU = 3.0
RHO = 2.0
NU = MU / RHO


def build(mesh, wall='bottom', **kwargs):
    return KEpsilonWallFunction(mesh, mu=MU, rho=RHO, nu=NU, C_mu=0.09,
                                kappa=0.41, E=9.8, wall_boundary=wall, **kwargs)


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


def test_first_cell_mask_matches_the_full_expanded_wall_layer():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    values = wf.first_cell_mask().vec.FV().NumPy()
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


def test_epsilon_wall_uses_high_re_equilibrium_relation():
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    k = 0.4
    epsilon = ngs.GridFunction(wf._fes0)
    epsilon.Set(wf.epsilon_wall_cell(ngs.CoefficientFunction(k)))
    distance = wf._epsilon_dist_cell.vec.FV().NumPy()
    expected = 0.09 ** 0.75 * k ** 1.5 / (0.41 * distance)
    assert epsilon.vec.FV().NumPy() == pytest.approx(expected, rel=1e-9)


def test_marked_cells_are_the_closest_cells_to_the_wall():
    mesh = ngs.Mesh(CHANNEL)
    wf = build(mesh, 'wall')
    dist = wf.wall_distance_cell().vec.FV().NumPy()
    marked = wf.mask.vec.FV().NumPy() > 0.5
    assert dist[marked].max() < dist[~marked].max()


# ----------------------------------------------------------------------
# Tangential traction
# ----------------------------------------------------------------------

def set_velocity(mesh, cf):
    gfu = ngs.GridFunction(ngs.VectorH1(mesh, order=2))
    gfu.Set(cf)
    return gfu


def tau_on_marked(wf):
    values = wf.tau_cell.vec.FV().NumPy()
    return values[wf.first_cell_mask().vec.FV().NumPy() > 0.5]


def test_manufactured_tangential_traction_gives_its_known_magnitude():
    """u = (y, 0) on the y = 0 wall: sigma_xy = mu * du_x/dy = mu, purely
    tangential, so tau_wall == mu."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    wf.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.y, 0.0))))
    assert tau_on_marked(wf) == pytest.approx(MU, rel=1e-9)


def test_pure_normal_traction_gives_zero_tangential_shear():
    """u = (x, -y) is divergence free with traction purely normal on a flat
    y = 0 wall, so the tangential projection must annihilate it."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    wf.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.x, -ngs.y))))
    assert tau_on_marked(wf).max() < 1e-9


def test_normal_traction_does_not_leak_into_the_shear():
    """Superposing a normal-traction field on a shear field must leave tau
    unchanged -- this is what the legacy |sigma.n| got wrong."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')

    wf.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.y, 0.0))))
    shear_only = tau_on_marked(wf).copy()

    wf.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.y + ngs.x, -ngs.y))))
    combined = tau_on_marked(wf)

    assert combined == pytest.approx(shear_only, rel=1e-9)


def test_shear_is_invariant_under_rotation_of_wall_and_velocity():
    """The whole point of the change: rotate the geometry and the velocity field
    together and the wall shear is unchanged. The legacy x-binning could not do
    this. 'left' is the y-axis wall; u = (0, x) shears along it exactly as
    u = (y, 0) shears along 'bottom'."""
    mesh = ngs.Mesh(SQUARE)

    horizontal = build(mesh, 'bottom')
    horizontal.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.y, 0.0))))

    rotated = build(mesh, 'left')
    rotated.update(set_velocity(mesh, ngs.CoefficientFunction((0.0, ngs.x))))

    assert tau_on_marked(horizontal) == pytest.approx(MU, rel=1e-9)
    assert tau_on_marked(rotated) == pytest.approx(MU, rel=1e-9)


def test_tau_is_floored_positive_before_the_first_update():
    """eval_nu_t must be safe (sqrt, log) even if update() has not run."""
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    assert wf.tau_cell.vec.FV().NumPy().min() > 0.0


def test_zero_velocity_keeps_tau_positive():
    mesh = ngs.Mesh(SQUARE)
    wf = build(mesh, 'bottom')
    wf.update(set_velocity(mesh, ngs.CoefficientFunction((0.0, 0.0))))
    assert wf.tau_cell.vec.FV().NumPy().min() > 0.0


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
    assert nu_t[marked] == pytest.approx(0.0, abs=1e-12)


def test_log_range_yplus_selects_the_log_law(channel_wf):
    mesh, wf = channel_wf
    yplus = 50.0
    marked, k = force_yplus(wf, yplus)
    nu_t = sample_nu_t(wf, mesh, k=k, eps=1.0)
    expected = NU * (0.41 * yplus - 1)
    assert expected > 0
    assert nu_t[marked] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('threshold', [11.25, 30.0])
def test_wall_viscosity_is_continuous_at_transition_thresholds(channel_wf, threshold):
    mesh, wf = channel_wf
    samples = []
    for yplus in (threshold - 1e-5, threshold + 1e-5):
        marked, k = force_yplus(wf, yplus)
        values = sample_nu_t(wf, mesh, k=k, eps=wf.epsilon_wall_cell(k))
        samples.append(values[marked])

    scale = np.maximum(np.maximum(np.abs(samples[0]), np.abs(samples[1])), 1.0)
    assert np.max(np.abs(samples[1] - samples[0]) / scale) < 1e-4


def test_high_yplus_remains_on_wall_law_inside_the_mask(channel_wf):
    mesh, wf = channel_wf
    yplus = 500.0
    marked, k = force_yplus(wf, yplus)
    eps = 2.5
    nu_t = sample_nu_t(wf, mesh, k=k, eps=eps)
    expected = NU * (0.41 * yplus - 1.0)
    assert nu_t[marked] == pytest.approx(expected, rel=1e-6)


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
    assert nu_t[marked] == pytest.approx(0.0, abs=1e-12)  # and the layer differs

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
    wf.update(set_velocity(mesh, ngs.CoefficientFunction((ngs.y * (1 - ngs.y), 0.0))))
    assert wf.tau_cell.vec.FV().NumPy().max() > 0
