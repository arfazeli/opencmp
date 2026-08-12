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

# need to check if User is running WSL
import os, platform

def _running_in_wsl():
    """Detect whether user is running WSL (any version)."""
    if 'WSL_DISTRO_NAME' in os.environ:
        return True
    
    if 'microsoft' in platform.uname().release.lower():
        return True

    try:
        with open('/proc/version') as f:
            return 'microsoft' in f.read().lower()
    except FileNotFoundError:
        return False

if _running_in_wsl():
    # WSLg's GPU driver stack is unstable with vtkSurfaceLICMapper / hardware
    # rendering -- forces software rendering to avoid crashing the whole
    # session. See: <link to whatever issue/notes you have on this>
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '450'

# PyVista is an optional dependency, so it is imported only when present
missing_pyvista = not can_import_module('pyvista')
if not missing_pyvista:
    import numpy as np
    import pyvista as pv
    import vtk

_COLORMAP = 'viridis'
_LIC_INTENSITY = 0.5
if _running_in_wsl():
    _LIC_ENHANCE_CONSTRAST = 0
else:
    _LIC_ENHANCE_CONSTRAST = 3
_RANGE_SAMPLES = 15

_BAR_ARGS = dict(width=0.6, height=0.12, position_x=0.2, position_y=0.04)


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
    frames_dir.mkdir(parents=True, exist_ok=True)

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

    files = _read_time_series(output_dir, model.name())
    mesh = pv.read(files[0][1])
    fields = _discover_fields(mesh, plot_variables, model.save_names)
    plotter, panels = _build_plotter(mesh, fields, vector_plot_mode, files)
    
    # frames loop
    for frame_num, (time, path) in enumerate(files):
        if frame_num > 0:
            new_mesh = pv.read(path)

            # mesh update
            for var in fields:
                mesh.point_data[var] = new_mesh.point_data[var]

            # vector helpers
            for var, is_vector in fields.items():
                if not is_vector:
                    continue
                data = mesh.point_data[var]
                mesh.point_data[f"{var}_magnitude"] = np.linalg.norm(data, axis=1)
                if vector_plot_mode.get(var, 'color') == 'lic':
                    if data.shape[1] == 2:
                        zeros = np.zeros(len(data))
                        mesh.point_data[f"{var}_3d"] = np.column_stack([data, zeros])
                    else:
                        mesh.point_data[f"{var}_3d"] = data
            
            mesh.Modified()

            # lic sync
            for var, panel in panels.items():
                if panel["kind"] == "lic":
                    surface = panel["surface"]
                    surface_point_ids = panel["surface_point_ids"]
                    surface.point_data[f"{var}_3d"] = mesh.point_data[f"{var}_3d"][surface_point_ids]
                    surface.point_data[f"{var}_magnitude"] = mesh.point_data[f"{var}_magnitude"][surface_point_ids]
                    surface.Modified()
            
        plotter.render()
        frame_path = frames_dir / f"{frame_num:04d}.png"
        plotter.screenshot(str(frame_path))

    plotter.close()

    # raise NotImplementedError('PyVista rendering is not implemented yet.')


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

    vtu_dir = output_dir / f"{model_name}_vtu"

    if not vtu_dir.is_dir():
        raise FileNotFoundError(
            f"No .vtu output directory found at '{vtu_dir}'."
        )
    
    vtu_files = list(vtu_dir.glob("*.vtu"))
    if not vtu_files:
        raise FileNotFoundError(
            f"No .vtu files found in '{vtu_dir}'."
        )

    time_series = []
    for path in vtu_files:
        time_str = path.stem.split('_')[-1]
        time = float(time_str)
        time_series.append((time, path))
    
    time_series.sort(key=lambda pair: pair[0])

    return time_series

    # raise NotImplementedError


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

    fields: Dict[str, bool] = {}

    for var in plot_variables:
        if var not in save_names:
            available = ', '.join(sorted(save_names))
            raise ValueError(
                f'Requested plot variable "{var}" was not found among the saved variables. Available variables: {available}'
            )
        
        data = mesh.point_data[var]
        is_vector = data.ndim > 1 and data.shape[1] in (2, 3)
        fields[var] = is_vector

    return fields

    # raise NotImplementedError


def _build_plotter(mesh, fields: Dict[str, bool],
                   vector_plot_mode: Dict[str, str],
                   files: List[Tuple[float, Path]]):
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

    # colour ranges
    n_frames = len(files)
    sample_idxs = np.unique(np.linspace(0, n_frames - 1, min(_RANGE_SAMPLES, n_frames)).astype(int))

    colour_ranges = {var: [np.inf, -np.inf] for var in fields}
    for idx in sample_idxs:
        _, path = files[idx]
        sample_mesh = pv.read(path)
        for var, is_vector in fields.items():
            data = sample_mesh.point_data[var]
            vals = np.linalg.norm(data, axis=1) if is_vector else data
            colour_ranges[var][0] = min(colour_ranges[var][0], float(vals.min()))
            colour_ranges[var][1] = max(colour_ranges[var][1], float(vals.max()))
    colour_ranges = {var: (lo, hi) for var, (lo, hi) in colour_ranges.items()}

    # vector helpers
    for var, is_vector in fields.items():
        if not is_vector:
            continue
        data = mesh.point_data[var]
        mesh.point_data[f"{var}_magnitude"] = np.linalg.norm(data, axis=1)
        if vector_plot_mode.get(var, 'color') == 'lic':
            if data.shape[1] == 2:
                zeros = np.zeros(len(data))
                mesh.point_data[f"{var}_3d"] = np.column_stack([data, zeros])
            else:
                mesh.point_data[f"{var}_3d"] = data
    
    variables = list(fields)
    plotter = pv.Plotter(shape=(len(variables), 1), off_screen=True)
    panels: Dict[str, dict] = {}

    for row, var in enumerate(variables):
        plotter.subplot(row, 0)
        is_vector = fields[var]
        colour_range = colour_ranges[var]

        # scalar
        if not is_vector:
            actor = plotter.add_mesh(mesh, scalars=var, cmap=_COLORMAP, clim=colour_range, show_scalar_bar=False)
            lut_for_bar = actor.mapper.lookup_table
            plotter.add_text(var, position='upper_edge', font_size=10)
            panels[var] = {'kind': 'scalar'}
        else:
            mode = vector_plot_mode.get(var, 'color')

            # color
            if mode == 'color':
                actor = plotter.add_mesh(mesh, scalars=f"{var}_magnitude", cmap=_COLORMAP, clim=colour_range, show_scalar_bar=False)
                lut_for_bar = actor.mapper.lookup_table
                plotter.add_text(var, position='upper_edge', font_size=10)
                panels[var] = {'kind': 'color'}

            # lic
            elif mode == 'lic':
                surface = mesh.extract_surface(algorithm='dataset_surface')
                surface_point_ids = surface['vtkOriginalPointIds']

                mapper = vtk.vtkSurfaceLICMapper()
                mapper.SetInputArrayToProcess(
                    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, f'{var}_3d'
                )
                lic_interface = mapper.GetLICInterface()
                lic_interface.SetLICIntensity(_LIC_INTENSITY)
                lic_interface.SetEnhanceContrast(_LIC_ENHANCE_CONSTRAST)
                mapper.SetInputData(surface)

                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(f'{var}_magnitude')
                lut_for_bar = pv.LookupTable(_COLORMAP, n_values=256)
                lut_for_bar.scalar_range = colour_range
                mapper.SetLookupTable(lut_for_bar)
                mapper.SetUseLookupTableScalarRange(True)
                mapper.ScalarVisibilityOn()

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                plotter.add_actor(actor)
                plotter.add_text(f'{var} (LIC)', position='upper_edge', font_size=10)

                panels[var] = {
                    'kind': 'lic',
                    'surface': surface,
                    'surface_point_ids': surface_point_ids
                }

            else:
                raise ValueError(
                    f'Unrecognized vector_plot_mode "{mode}" for variable "{var}"./nExpected "color" or "lic".'
                )

        # scalar bar
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut_for_bar)
        bar.SetTitle(var)
        bar.SetOrientationToHorizontal()
        bar.SetWidth(_BAR_ARGS['width'])
        bar.SetHeight(_BAR_ARGS['height'])
        bar.SetPosition(_BAR_ARGS['position_x'], _BAR_ARGS['position_y'])

        bar.UnconstrainedFontSizeOn()
        for prop in (bar.GetLabelTextProperty(), bar.GetTitleTextProperty()):
            prop.SetColor(0, 0, 0)
            prop.SetBold(False)
            prop.ShadowOff()
        bar.GetLabelTextProperty().SetFontSize(16)
        bar.GetTitleTextProperty().SetFontSize(20)

        plotter.add_actor(bar)
        plotter.view_xy()
    
    return plotter, panels

    # raise NotImplementedError