"""Structural regression tests for the k-epsilon INS model."""

import inspect
import numpy as np
import re
from types import SimpleNamespace

import ngsolve as ngs
import pytest
from netgen.geom2d import unit_square

from opencmp.helpers.limiter import Limiter
from opencmp.helpers.wall_func import KEpsilonWallFunction
from opencmp.models import KEpsilonINS, models_dict
from opencmp.models.ins import INS


def _auto_inlet_model(explicit_ic=(), explicit_bc=()) -> KEpsilonINS:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    model = object.__new__(KEpsilonINS)
    model.mesh = mesh
    model.auto_turbulence_inlet = 'left'
    model.turbulence_hydraulic_diameter = 2.0
    model.turbulence_length_scale_ratio = 0.07
    model.kv = [0.1]
    model.C_mu = 0.09
    model.t_param = [ngs.Parameter(0.0)]
    model.model_components_ic = {'u': 0, 'p': 1, 'k': 2, 'epsilon': 3}
    model.BC = {'dirichlet': {
        'u': {'left': [ngs.CoefficientFunction((1.0, 0.0))]},
    }}
    for component in explicit_bc:
        model.BC['dirichlet'][component] = {'left': [9.0]}
    model.dirichlet_names = {'u': 'left'}
    model.ic_functions = SimpleNamespace(ic_dict={
        model.name(): {component: {'all': [9.0]} for component in explicit_ic}
    })
    spaces = [ngs.L2(mesh, order=0) for _ in range(4)]
    model.IC = ngs.GridFunction(ngs.FESpace(spaces))
    for component in explicit_ic:
        model.IC.components[model.model_components_ic[component]].Set(9.0)
    return model


def _turbulence_stub(k_value: float, epsilon_value: float,
                     ratio: float = 2000.0) -> KEpsilonINS:
    """Bare model carrying only what _regularized_turbulence reads."""
    model = object.__new__(KEpsilonINS)
    model.model_components = {'u': 0, 'p': 1, 'k': 2, 'epsilon': 3}
    model._bounded = False
    model.k_floor = 1e-8
    model.epsilon_floor = 1e-4
    model.C_mu = 0.09
    model.kv = [2e-5]
    model.max_viscosity_ratio = ratio
    model.UIter = SimpleNamespace(components=[
        None, None,
        ngs.CoefficientFunction(k_value),
        ngs.CoefficientFunction(epsilon_value)])
    model._k_cell = ngs.CoefficientFunction(k_value)
    model._epsilon_cell = ngs.CoefficientFunction(epsilon_value)
    # The stub carries no velocity, so the strain-based realizability bound
    # cannot be evaluated; test_realizability_* builds its own model for that.
    model.realizability_limiter = False
    return model


def test_k_epsilon_ins_is_registered_as_an_ins_model() -> None:
    assert models_dict['KEpsilonINS'] is KEpsilonINS
    assert issubclass(KEpsilonINS, INS)


def test_k_epsilon_ins_component_contract() -> None:
    model = object.__new__(KEpsilonINS)

    assert model._define_model_components() == {
        'u': 0,
        'p': 1,
        'k': 2,
        'epsilon': 3,
    }
    assert model._define_time_derivative_components() == [{
        'u': True,
        'p': False,
        'k': True,
        'epsilon': True,
    }]


def test_k_epsilon_supports_scalar_neumann_boundaries() -> None:
    model = object.__new__(KEpsilonINS)

    assert 'neumann' in model._define_bc_types()


def test_k_epsilon_reuses_cached_turbulent_viscosity() -> None:
    model = object.__new__(KEpsilonINS)
    cached_viscosity = object()
    model._turbulent_viscosity = [cached_viscosity]

    assert model._get_turbulent_viscosity(0) is cached_viscosity


def test_epsilon_bound_caps_the_eddy_viscosity_ratio() -> None:
    """The epsilon bound keeps nu_t/nu below the configured ratio."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    k, epsilon = _turbulence_stub(0.0149, 1e-4)._regularized_turbulence(0)
    ratio = ngs.Integrate(0.09 * k ** 2 / epsilon, mesh) / 2e-5
    assert ratio == pytest.approx(2000.0, rel=1e-9)

    k, epsilon = _turbulence_stub(0.0149, 3e-3)._regularized_turbulence(0)
    assert ngs.Integrate(epsilon, mesh) == pytest.approx(3e-3, rel=1e-9)


def test_epsilon_k_ratio_is_smoothly_bounded() -> None:
    model = object.__new__(KEpsilonINS)
    model.max_epsilon_k_ratio = 10.0
    model._bounded = False

    regular = model._epsilon_k_ratio(1.0, 0.1)
    floor_state = model._epsilon_k_ratio(1e-8, 1e-4)

    assert regular == pytest.approx(0.1 / 1.01)
    assert floor_state < 10.0
    assert floor_state == pytest.approx(1e-4 / 1.001e-5)


def test_ins_viscosity_hook_preserves_laminar_behavior() -> None:
    model = object.__new__(INS)
    model.kv = [1.25, 2.5]

    assert model._get_effective_viscosity(0) == 1.25
    assert model._get_effective_viscosity(1) == 2.5


def test_turbulence_constants_fall_back_to_standard_values() -> None:
    """Omitting a constant from [PARAMETERS] must use the prescribed value."""
    model = object.__new__(KEpsilonINS)

    for name, default in KEpsilonINS.DEFAULT_PARAMETERS.items():
        assert model._parameter({}, name) == default
    assert model._parameter({'c_mu': {'all': [0.1, 0.1]}}, 'c_mu') == 0.1


def test_every_constant_the_model_reads_has_a_default() -> None:
    """_set_model_parameters must not KeyError on a config that omits constants."""
    source = inspect.getsource(KEpsilonINS._set_model_parameters)
    requested = set(re.findall(r"_parameter\(\s*parameters,\s*'([a-z_]+)'", source))

    assert requested, 'parameter reads no longer match the expected pattern'
    assert requested <= set(KEpsilonINS.DEFAULT_PARAMETERS)


def test_k_based_wall_friction_velocity_remains_the_default() -> None:
    assert KEpsilonINS.DEFAULT_PARAMETERS['wall_u_tau_method'] == 0.0


def test_auto_turbulence_defaults_use_inlet_bulk_velocity() -> None:
    model = _auto_inlet_model()
    model._apply_auto_turbulence_defaults()

    reynolds = 1.0 * 2.0 / 0.1
    intensity = 0.16 * reynolds ** (-1.0 / 8.0)
    expected_k = 1.5 * intensity ** 2
    expected_epsilon = (0.09 ** 0.75 * expected_k ** 1.5
                        / (0.07 * 2.0))

    assert model.BC['dirichlet']['k']['left'] == pytest.approx([expected_k])
    assert model.BC['dirichlet']['epsilon']['left'] == pytest.approx(
        [expected_epsilon])
    assert model.dirichlet_names['k'] == 'left'
    assert model.dirichlet_names['epsilon'] == 'left'
    assert np.asarray(model.IC.components[2].vec).mean() == pytest.approx(
        expected_k)
    assert np.asarray(model.IC.components[3].vec).mean() == pytest.approx(
        expected_epsilon)


def test_auto_turbulence_defaults_preserve_explicit_values() -> None:
    model = _auto_inlet_model(
        explicit_ic=('k', 'epsilon'), explicit_bc=('k', 'epsilon'))
    model._apply_auto_turbulence_defaults()

    assert model.BC['dirichlet']['k']['left'] == [9.0]
    assert model.BC['dirichlet']['epsilon']['left'] == [9.0]
    assert np.asarray(model.IC.components[2].vec).mean() == pytest.approx(9.0)
    assert np.asarray(model.IC.components[3].vec).mean() == pytest.approx(9.0)


def test_auto_turbulence_defaults_reject_non_inward_flux() -> None:
    model = _auto_inlet_model()
    model.BC['dirichlet']['u']['left'] = [
        ngs.CoefficientFunction((-1.0, 0.0))]

    with pytest.raises(ValueError, match='nonzero inward net flux'):
        model._apply_auto_turbulence_defaults()


def test_k_epsilon_wall_function_is_enabled_by_default() -> None:
    class EmptyConfig:
        def get_item(self, *_args, **_kwargs):
            raise KeyError

    model = object.__new__(KEpsilonINS)
    model.config = EmptyConfig()
    model._pre_init()

    assert model.wall_function is True
    assert model.wall_boundary == 'wall'


def test_pre_init_sets_every_documented_option() -> None:
    """_pre_init drives itself from DEFAULT_OPTIONS, so the dict is the contract."""
    class EmptyConfig:
        def get_item(self, *_args, **_kwargs):
            raise KeyError

    model = object.__new__(KEpsilonINS)
    model.config = EmptyConfig()
    model._pre_init()

    for key, default in KEpsilonINS.DEFAULT_OPTIONS.items():
        assert getattr(model, key) == default


def _limiter_stub(bounded: bool) -> KEpsilonINS:
    """Bare model carrying only what _regularized_turbulence reads."""
    model = _turbulence_stub(0.5, 1.0)
    model._bounded = bounded
    return model


def test_recovered_coefficient_floors_apply_when_solution_is_bounded() -> None:
    """Recovered closure fields retain a final coefficient-level safeguard."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    model = _limiter_stub(True)
    model._k_cell = ngs.CoefficientFunction(-1.0)

    k, _ = model._regularized_turbulence(0)
    assert ngs.Integrate(k, mesh) == pytest.approx(model.k_floor, rel=1e-9)


def test_coefficient_floors_still_apply_on_unbounded_spaces() -> None:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    model = _limiter_stub(False)
    model._k_cell = ngs.CoefficientFunction(-1.0)

    k, _ = model._regularized_turbulence(0)
    assert ngs.Integrate(k, mesh) == pytest.approx(model.k_floor, rel=1e-9)


def test_bound_epsilon_applies_in_both_modes() -> None:
    """boundEpsilon depends on the local k, so no limiter can impose it; it must
    survive whether or not the slope limiter is on."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    for bounded in (True, False):
        model = _limiter_stub(bounded)
        _, epsilon = model._regularized_turbulence(0)
        ratio = ngs.Integrate(0.09 * 0.5 ** 2 / epsilon, mesh) / 2e-5
        assert ratio <= 2000.0 + 1e-6, bounded


def test_bezier_bound_holds_k_above_its_floor_inside_the_element() -> None:
    """What the dropped coefficient guards relied on: the limited polynomial is above
    the floor everywhere in the element, not just at the DOFs."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = ngs.L2(mesh, order=1)
    field = ngs.GridFunction(fes)
    field.Set(ngs.x - 0.5)                      # negative over half the domain
    floor = 1e-8

    Limiter(mesh).bezier_bound(field, fes, fes.globalorder, (floor, 1e20))

    # Sample the limited polynomial densely rather than trusting the DOFs.
    below = ngs.Integrate(ngs.IfPos(floor - field, 1.0, 0.0), mesh)
    assert below == pytest.approx(0.0, abs=1e-12)


def test_floored_epsilon_does_not_inflate_bulk_viscosity_to_the_cap() -> None:
    """A bounded epsilon sitting at its floor is a clamped undershoot, not physics.

    Trusting C_mu*k**2/epsilon there saturates nu_t at the viscosity cap and paints
    cap-to-zero jumps against neighbouring cells (the backward-facing-step scar).
    The bulk viscosity must fade out instead.
    """
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    model = _turbulence_stub(2.6e-3, 1e-8)   # moderate k, epsilon at its floor
    model._bounded = True
    model._wallf = None
    model.epsilon_floor = 1e-8

    nu_t = ngs.Integrate(model._build_turbulent_viscosity(0), mesh)
    cap = model.max_viscosity_ratio * model.kv[0]
    assert nu_t < 0.05 * cap

    # A healthy epsilon must be essentially untouched by the trust factor.
    model = _turbulence_stub(2.6e-3, 3e-4)
    model._bounded = True
    model._wallf = None
    model.epsilon_floor = 1e-8
    nu_t = ngs.Integrate(model._build_turbulent_viscosity(0), mesh)
    expected = 0.09 * 2.6e-3 ** 2 / 3e-4
    assert nu_t == pytest.approx(expected, rel=0.01)


def _realizability_model(k_value: float, epsilon_value: float,
                         mesh: ngs.Mesh, a: float = 1.0) -> KEpsilonINS:
    """Stub with a real velocity field so the strain-rate bound can be evaluated."""
    model = _turbulence_stub(k_value, epsilon_value)
    model.realizability_limiter = True
    model.realizability_coefficient = a
    model._wallf = None
    model._bounded = False
    fes = ngs.VectorH1(mesh, order=1)
    u = ngs.GridFunction(fes)
    u.Set(ngs.CoefficientFunction((ngs.y, 0.0)))   # grad u = [[0,1],[0,0]] -> S = 1
    model.UIter = SimpleNamespace(components=[u, None,
                                              ngs.CoefficientFunction(k_value),
                                              ngs.CoefficientFunction(epsilon_value)])
    return model


def test_realizability_bound_caps_nu_t_by_the_strain_rate() -> None:
    """Durbin: nu_t <= a*k/(sqrt(6)*S). With u=(y,0) the strain magnitude S is 1,
    so a collapsing epsilon must not drive nu_t past a*k/sqrt(6)."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    k_value = 1e-2
    model = _realizability_model(k_value, 1e-12, mesh)   # epsilon -> 0
    area = ngs.Integrate(ngs.CoefficientFunction(1.0), mesh)
    nu_t = ngs.Integrate(model._build_turbulent_viscosity(0), mesh) / area

    expected = k_value / np.sqrt(6.0)
    assert nu_t == pytest.approx(expected, rel=1e-6)

    # Without the bound this same state saturates the viscosity-ratio cap,
    # which is the 0-to-cap jump that wrecks the high-order solve.
    unlimited = _realizability_model(k_value, 1e-12, mesh)
    unlimited.realizability_limiter = False
    cap = unlimited.max_viscosity_ratio * unlimited.kv[0]
    nu_t_unlimited = ngs.Integrate(
        unlimited._build_turbulent_viscosity(0), mesh) / area

    assert nu_t_unlimited == pytest.approx(cap, rel=1e-6)
    assert nu_t < nu_t_unlimited


def test_realizability_bound_is_inactive_when_bulk_nu_t_is_small() -> None:
    """The bound is a max-of-two min; a healthy k-epsilon state must pass through."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    k_value, epsilon_value = 1e-3, 1.0        # C_mu k^2/eps = 9e-11, tiny
    model = _realizability_model(k_value, epsilon_value, mesh)
    area = ngs.Integrate(ngs.CoefficientFunction(1.0), mesh)
    nu_t = ngs.Integrate(model._build_turbulent_viscosity(0), mesh) / area

    assert nu_t == pytest.approx(0.09 * k_value ** 2 / epsilon_value, rel=1e-6)
