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
PyVista rendering of saved simulation results.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from ..config_functions import ConfigParser
from ..helpers.misc import can_import_module
from ..models import Model

# PyVista is an optional dependency, so it is imported only when present
missing_pyvista = not can_import_module('pyvista')
if not missing_pyvista:
    import pyvista as pv


def visualize_results(config_parser: ConfigParser, model: Model) -> None:
    """Render the configured variables from the saved .vtu files to PNG frames.

    Args:
        config_parser: The config parser for the simulation being post-processed.
        model: The model that produced the results. Used for ``mesh.dim``,
            ``name()`` (to locate the output directory) and ``save_names``
            (to validate the requested variables).
    """
    if missing_pyvista:
        raise ImportError('pyvista module is not installed. Install it with `pip install pyvista`.')

    if model.mesh.dim != 2:
        print('Plotting is only supported for 2D simulations; skipping.')
        return

    plot_variables = config_parser.get_list(['VISUALIZATION', 'plot_variables'], str)

    if not plot_variables:
        print('generate_plots is True but plot_variables is empty; nothing to plot.')
        return

    # '->' syntax, parsed by ConfigParser.get_dict. all_str keeps the values as
    # the raw mode strings ('lic'/'color') instead of trying to evaluate them.
    vector_plot_mode = config_parser.get_dict(
        ['VISUALIZATION', 'vector_plot_mode'], model.run_dir, all_str=True)

    run_dir = Path(config_parser.get_item(['OTHER', 'run_dir'], str))
    output_dir = run_dir / 'output'
    frames_dir = output_dir / 'frames'

    # TODO: everything below is the prototype port.
    #
    #   1. files = _read_time_series(output_dir, model.name())
    #   2. fields = _discover_fields(pv.read(files[0][1]), plot_variables, model.save_names)
    #   3. plotter = _build_plotter(mesh, fields, vector_plot_mode)
    #   4. loop the files in time order, update the plotter, screenshot each frame
    #      into frames_dir as 0000.png, 0001.png, ...
    #
    # A stationary run has a single .vtu, so the loop simply runs once and
    # produces 0000.png -- no special-casing needed.
    raise NotImplementedError('PyVista rendering is not implemented yet.')


def _read_time_series(output_dir: Path, model_name: str) -> List[Tuple[float, Path]]:
    """Return the saved .vtu files paired with their simulation times, in time order.

    Scans ``<output_dir>/<model_name>_vtu/``. ``sol_to_vtu`` names each file
    ``<model_name>_<time>.vtu`` (the name is carried over from the source .sol),
    so the time is recoverable from the filename alone.
    Take the stem, split on ``_`` and ``float()`` the last field -- the same
    parse ``sol_to_vtu`` itself uses to build the name.

    Sort **numerically on the parsed float**, never on the filename string.
    Times are written in whatever repr Python produced, so a lexicographic sort
    misorders both scientific notation and multi-digit times::

        lexicographic: 0.0002, 1.953125e-08, 10.0, 9.0     # wrong
        numeric:       1.953125e-08, 0.0002, 9.0, 10.0     # right

    A stationary run leaves a single file, which needs no special handling.

    Args:
        output_dir: The ``<run_dir>/output`` directory.
        model_name: ``model.name()``, used to build the .vtu subdirectory name.

    Returns:
        ``(time, path)`` pairs sorted by increasing time.

    Raises:
        FileNotFoundError: If the .vtu directory is missing or holds no .vtu
            files -- meaning the run did not save any output to convert.
    """
    # TODO: port from the prototype.
    raise NotImplementedError


def _discover_fields(mesh, plot_variables: List[str],
                     save_names: List[str]) -> Dict[str, bool]:
    """Validate the requested variables and classify each as scalar or vector.

    Classification comes from the shape of the point-data array on the mesh, not
    from the model definition, so it also covers the 2-vs-3 component question
    for 2D vector fields (NGSolve's ``VTKOutput`` may write either).

    Args:
        mesh: A PyVista mesh read from one of the .vtu files.
        plot_variables: The variable names requested in the config file.
        save_names: ``model.save_names`` -- the names actually present in the
            .vtu point data.

    Returns:
        Maps each requested variable name to True if it is a vector field,
        False if it is a scalar.

    Raises:
        ValueError: If a requested variable does not exist, listing the
            available names so the user can correct the config file.
    """
    # TODO: port from the prototype (its `discover_fields` / `vector3d` logic).
    raise NotImplementedError


def _build_plotter(mesh, fields: Dict[str, bool],
                   vector_plot_mode: Dict[str, str]):
    """Build the off-screen stacked-panel plotter, one panel per variable.

    Panel type per variable:

    * scalar             -> ``add_mesh`` coloured by the scalar
    * vector, ``color``  -> ``add_mesh`` coloured by the vector magnitude
    * vector, ``lic``    -> ``vtkSurfaceLICMapper`` flow texture

    Colour ranges are sampled once ("auto") and then held fixed across all
    frames, so the colours mean the same thing in every frame of the sequence.

    Extract the surface **once** and reuse it for every LIC panel -- re-extracting
    per panel or per frame is what made the prototype slow.

    Args:
        mesh: A PyVista mesh read from the first .vtu file, used to set up the
            panels and sample the colour ranges.
        fields: Output of :func:`_discover_fields`.
        vector_plot_mode: The ``vector_plot_mode`` config dict. Vector variables
            absent from it default to ``color``.

    Returns:
        A ``pyvista.Plotter`` with ``off_screen=True``, ready to be updated per
        frame and screenshotted.
    """
    # TODO: port from the prototype.
    raise NotImplementedError
