"""Structural regression tests for the k-epsilon INS model."""

from opencmp.models import KEpsilonINS, models_dict
from opencmp.models.ins import INS


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


def test_k_epsilon_slope_limiter_is_disabled_by_default() -> None:
    class EmptyConfig:
        def get_item(self, *_args, **_kwargs):
            raise KeyError

    model = object.__new__(KEpsilonINS)
    model.config = EmptyConfig()
    model._pre_init()

    assert model.slope_limiter is False


def test_k_epsilon_slope_limiter_can_be_enabled_explicitly() -> None:
    class SlopeLimiterConfig:
        def get_item(self, keys, *_args, **_kwargs):
            if keys == ['OTHER', 'slope_limiter']:
                return True
            raise KeyError

    model = object.__new__(KEpsilonINS)
    model.config = SlopeLimiterConfig()
    model._pre_init()

    assert model.slope_limiter is True


def test_k_epsilon_reuses_cached_turbulent_viscosity() -> None:
    model = object.__new__(KEpsilonINS)
    cached_viscosity = object()
    model._turbulent_viscosity = [cached_viscosity]

    assert model._get_turbulent_viscosity(0) is cached_viscosity


def test_epsilon_wall_target_is_change_limited_and_relaxed() -> None:
    value, limited = KEpsilonINS._relax_epsilon_wall_value(
        raw_target=10.0, previous=1.0, relaxation=0.15,
        change_factor=2.0)

    assert limited is True
    assert value == 1.15


def test_epsilon_wall_target_accepts_unlimited_change() -> None:
    value, limited = KEpsilonINS._relax_epsilon_wall_value(
        raw_target=1.5, previous=1.0, relaxation=0.2,
        change_factor=2.0)

    assert limited is False
    assert value == 1.1


def test_ins_viscosity_hook_preserves_laminar_behavior() -> None:
    model = object.__new__(INS)
    model.kv = [1.25, 2.5]

    assert model._get_kinematic_viscosity(0) == 1.25
    assert model._get_kinematic_viscosity(1) == 2.5
