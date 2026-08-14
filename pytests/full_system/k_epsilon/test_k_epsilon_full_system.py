"""Full-system smoke coverage for the single-phase k-epsilon model."""

from opencmp.config_functions import ConfigParser
from opencmp.helpers.testing import run_example


def test_transient_cg_smoke() -> None:
    config = ConfigParser('pytests/full_system/k_epsilon/config')
    run_example(config)


def test_transient_dg_smoke() -> None:
    config = ConfigParser('pytests/full_system/k_epsilon/config')
    config['DG']['DG'] = 'True'
    config['FINITE ELEMENT SPACE']['elements'] = (
        'u -> HDiv\n'
        'p -> L2\n'
        'k -> L2\n'
        'epsilon -> L2'
    )
    run_example(config)


def test_wall_function_smoke() -> None:
    config = ConfigParser('pytests/full_system/k_epsilon/config')
    config['DG']['DG'] = 'True'
    config['FINITE ELEMENT SPACE']['elements'] = (
        'u -> HDiv\n'
        'p -> L2\n'
        'k -> L2\n'
        'epsilon -> L2'
    )
    config['OTHER']['wall_function'] = 'True'
    config['OTHER']['wall_boundary'] = 'bottom'
    run_example(config)
