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

"""Shared k-epsilon closure for single- and two-fluid flow models."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import ngsolve as ngs
from ngsolve import (BilinearForm, FESpace, GridFunction, LinearForm,
                     Parameter, Preconditioner)
from ngsolve.comp import ProxyFunction

from .ins import INS
from .tfm import TwoFluidModel
from ..helpers.dg import avg, jump, weighted_grad_avg
from ..helpers.limiter import Limiter
from ..helpers.math import Max
from ..helpers.ngsolve_ import get_special_functions
from ..helpers.wall_func import KEpsilonWallFunction


class KEpsilonModel:
    """Reusable standard high-Reynolds-number k-epsilon closure.

    Concrete models combine this mixin with a momentum model and identify the
    velocity carrying turbulence through ``TURBULENCE_VELOCITY``.
    """

    TURBULENCE_VELOCITY = 'u'

    DEFAULT_PARAMETERS = {
        'c_mu': 0.09,
        'c_1': 1.44,
        'c_2': 1.92,
        'sigma_k': 1.0,
        'sigma_epsilon': 1.3,
        'kappa': 0.4187,
        'e_log': 9.793,
        'k_floor': 1e-10,
        'epsilon_floor': 1e-10,
        'max_viscosity_ratio': 1e5,
        'production_limit_coefficient': 10.0,
        'max_epsilon_k_ratio': 10.0,
        'realizability_coefficient': 1.0,
        'wall_u_tau_method': 0.0,
        'turbulence_hydraulic_diameter': 0.0,
        'turbulence_length_scale_ratio': 0.07,
    }

    #: Optional ``[OTHER]`` switches
    DEFAULT_OPTIONS = {
        'wall_function': True,
        'wall_boundary': 'wall',
        'production_limiter': True,
        'realizability_limiter': True,
        # Write per-Picard-iteration closure diagnostics and selected VTU files.
        'k_epsilon_diagnostics': False,
        'k_epsilon_diagnostic_vtu_start': 3,
        'k_epsilon_diagnostic_vtu_end': 10,
        # Boundary marker whose prescribed velocity generates default inlet and
        # initial k-epsilon data. Empty means that all values remain user supplied.
        'auto_turbulence_inlet': '',
    }

    def _parameter(self, parameters: Dict, name: str) -> float:
        """Config value if present, else the standard constant from DEFAULT_PARAMETERS.

        These are all constants, so the per-time-level list the config parser
        returns is collapsed to its first entry.
        """
        if name not in parameters:
            return self.DEFAULT_PARAMETERS[name]
        return parameters[name]['all'][0]

    def _optional_config(self, key: str):
        """One optional [OTHER] switch, from DEFAULT_OPTIONS if the config omits it."""
        default = self.DEFAULT_OPTIONS[key]
        try:
            value = self.config.get_item(['OTHER', key], type(default), quiet=True)
        except Exception:
            return default
        return default if value is None else value

    def _pre_init(self) -> None:
        super()._pre_init()
        for key in self.DEFAULT_OPTIONS:
            setattr(self, key, self._optional_config(key))

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        components = super()._define_model_components()
        next_index = max(index for index in components.values()
                         if index is not None) + 1
        components.update({'k': next_index, 'epsilon': next_index + 1})
        return components

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        components = super()._define_model_local_error_components()
        components.update({'k': True, 'epsilon': True})
        return components

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        components = super()._define_time_derivative_components()
        for form in components:
            form.update({'k': True, 'epsilon': True})
        return components

    def _define_bc_types(self) -> List[str]:
        return super()._define_bc_types() + ['neumann']

    def _construct_fes(self) -> FESpace:
        spaces = list(super()._construct_fes().components)
        scalar_order = max(self.interp_ord - 1, 0)

        turbulence_spaces = []
        for component in ('k', 'epsilon'):
            element = self.element[component]
            kwargs = {
                'mesh': self.mesh,
                'order': scalar_order,
                'dgjumps': self.DG,
            }
            if element != 'L2':
                kwargs['dirichlet'] = self.dirichlet_names.get(component, '')
            turbulence_spaces.append(getattr(ngs, element)(**kwargs))

        # TFM's optional pressure constraint is an internal trailing space and
        # is not represented in model_components. Keep k and epsilon before it.
        if getattr(self, 'mean_zero_pressure', False):
            spaces[-1:-1] = turbulence_spaces
        else:
            spaces.extend(turbulence_spaces)

        return FESpace(spaces, dgjumps=self.DG)


    def _bound_turbulence(self, gfu: GridFunction) -> None:
        """Floor and slope-limit the stored k and epsilon.

        Not optional: an unfloored k or epsilon zeroes the turbulent viscosity and
        with it the production term, an absorbing state with no way back above
        zero. Bezier bounds the DG polynomial everywhere (not just at sample
        nodes), so a positive cell mean can't hide a negative value inside the cell.
        """
        if self._limiter is None:
            return
        comp = self.model_components
        collect_diagnostics = getattr(self, 'k_epsilon_diagnostics', False)
        if collect_diagnostics and not hasattr(
                self, '_latest_limiter_diagnostics'):
            self._latest_limiter_diagnostics = {}
        for component, floor in (('k', self.k_floor),
                                 ('epsilon', self.epsilon_floor)):
            field = gfu.components[comp[component]]
            # Pass the stable FES, not the per-call component, so the limiter's
            # cache keys match across iterations.
            space = self.fes.components[comp[component]]
            if collect_diagnostics:
                raw = field.vec.FV().NumPy().copy()
                stats = self._limiter.bezier_bound(
                    field, space, space.globalorder, (floor, 1e20),
                    return_stats=True)
                correction = field.vec.FV().NumPy() - raw
                stats['correction_norm'] = float(np.linalg.norm(correction))
                self._latest_limiter_diagnostics[component] = stats
            else:
                self._limiter.bezier_bound(field, space, space.globalorder,
                                           (floor, 1e20))

    @staticmethod
    def _relax_component_on_free_dofs(current: GridFunction,
                                      previous: GridFunction, factor: float,
                                      space: FESpace) -> None:
        """Under-relax one component without changing constrained DOFs.

        The linear solve has already imposed the component's strong Dirichlet
        values. Blending the whole vector with the previous Picard iterate would
        reintroduce stale boundary values, so relaxation is restricted to the
        free-DOF mask of the component space.
        """
        current_values = current.vec.FV().NumPy()
        previous_values = previous.vec.FV().NumPy()
        free = np.fromiter(space.FreeDofs(), dtype=bool,
                           count=current_values.size)
        current_values[free] = (
            factor * current_values[free]
            + (1.0 - factor) * previous_values[free])

    def _set_model_parameters(self) -> None:
        super()._set_model_parameters()
        # Shared turbulence code uses ``kv`` for the molecular viscosity. TFM
        # exposes the same continuous-phase quantity as ``nu_c``.
        if not hasattr(self, 'kv'):
            self.kv = self.nu_c
        parameters = self.model_functions.model_parameters_dict
        self.C_mu = self._parameter(parameters, 'c_mu')
        self.C_1 = self._parameter(parameters, 'c_1')
        self.C_2 = self._parameter(parameters, 'c_2')
        self.sigma_k = self._parameter(parameters, 'sigma_k')
        self.sigma_epsilon = self._parameter(parameters, 'sigma_epsilon')
        self.kappa = self._parameter(parameters, 'kappa')
        self.E_log = self._parameter(parameters, 'e_log')

        # Floors and the viscosity ratio cap are numerical safeguards, not
        # replacements for physically meaningful ICs and BCs.
        self.k_floor = self._parameter(parameters, 'k_floor')
        self.epsilon_floor = self._parameter(parameters, 'epsilon_floor')
        self.max_viscosity_ratio = self._parameter(parameters, 'max_viscosity_ratio')
        self.production_limit_coefficient = self._parameter(
            parameters, 'production_limit_coefficient')
        self.max_epsilon_k_ratio = self._parameter(parameters, 'max_epsilon_k_ratio')
        self.realizability_coefficient = self._parameter(
            parameters, 'realizability_coefficient')
        self.wall_u_tau_method = int(self._parameter(
            parameters, 'wall_u_tau_method'))
        self.turbulence_hydraulic_diameter = self._parameter(
            parameters, 'turbulence_hydraulic_diameter')
        self.turbulence_length_scale_ratio = self._parameter(
            parameters, 'turbulence_length_scale_ratio')

        if self.production_limit_coefficient <= 0.0:
            raise ValueError('production_limit_coefficient must be positive.')
        if self.max_epsilon_k_ratio <= 0.0:
            raise ValueError('max_epsilon_k_ratio must be positive.')
        if self.wall_u_tau_method not in (0, 1):
            raise ValueError(
                'wall_u_tau_method must be 0 (k-based) or 1 (velocity-based).')
        if self.auto_turbulence_inlet:
            if self.turbulence_hydraulic_diameter <= 0.0:
                raise ValueError(
                    'turbulence_hydraulic_diameter must be positive when '
                    'auto_turbulence_inlet is enabled.')
            if self.turbulence_length_scale_ratio <= 0.0:
                raise ValueError(
                    'turbulence_length_scale_ratio must be positive when '
                    'auto_turbulence_inlet is enabled.')

    def _post_init(self) -> None:
        super()._post_init()
        if self.TURBULENCE_VELOCITY == 'u' and self.linearize != 'Oseen':
            raise NotImplementedError('KEpsilonINS currently supports Oseen linearization only.')

        if self.auto_turbulence_inlet:
            self._apply_auto_turbulence_defaults()

        self.UIter = ngs.GridFunction(self.fes)
        self.UIter.vec.data = self.IC.vec

        try:
            relaxation = self.config.get_list(['SOLVER', 'relaxation_factors'], float)
        except Exception:
            relaxation = []
        self.relaxation_factors = (
            relaxation if len(relaxation) == len(self.model_components)
            else [1.0] * len(self.model_components)
        )
        # Bounding needs a discontinuous L2 space; with CG/H1 turbulence spaces
        # the coefficient floors in _regularized_turbulence are the only safeguard.
        self._bounded = (self.DG and self.element['k'] == 'L2'
                         and self.element['epsilon'] == 'L2')
        self._limiter = Limiter(self.mesh) if self._bounded else None
        # The IC is user-supplied and may sit below the floors (e.g. zero fields).
        self._bound_turbulence(self.UIter)
        # nu_t is built straight from the DG solution, no H1 recovery. These are
        # live views into UIter, so they track it with no update step.
        self._k_cell = self.UIter.components[self.model_components['k']]
        self._epsilon_cell = self.UIter.components[self.model_components['epsilon']]
        self._wallf = None
        if self.wall_function:
            if not self.DG or self.element['epsilon'] != 'L2':
                raise NotImplementedError(
                    'The wall-law epsilon Dirichlet condition requires a '
                    'discontinuous L2 epsilon space.')
            self._wallf = KEpsilonWallFunction(
                self.mesh,
                nu=self.kv[0],
                C_mu=self.C_mu,
                kappa=self.kappa,
                E_log=self.E_log,
                wall_boundary=self.wall_boundary,
                u_tau_method=self.wall_u_tau_method,
            )
            self._update_wall_function()

        # Build and compile nu_t once per time level so every form reuses the
        # same optimized evaluation tree.
        self._turbulent_viscosity = []
        for time_step in range(len(self.t_param)):
            nu_t = self._build_turbulent_viscosity(time_step)
            self._turbulent_viscosity.append(nu_t.Compile())

        self._normal, _, self._penalty, _ = get_special_functions(self.mesh, self.nu)

        self._diagnostic_iteration = 0
        self._diagnostic_csv_initialized = False
        if self.k_epsilon_diagnostics:
            run_dir = self.config.get_item(['OTHER', 'run_dir'], str, quiet=True)
            self._diagnostic_dir = Path(run_dir) / 'output' / 'k_epsilon_picard_diagnostics'
            self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
            self._diagnostic_csv = self._diagnostic_dir / 'picard_iterations.csv'

    def _apply_auto_turbulence_defaults(self) -> None:
        """Fill missing inlet and initial k-epsilon data from inlet velocity.

        Explicit user values always take precedence. This first implementation
        supports one steady inlet marker.
        """
        marker = self.auto_turbulence_inlet
        velocity_markers = self.BC.get('dirichlet', {}).get(
            self.TURBULENCE_VELOCITY, {})
        if marker not in velocity_markers:
            raise ValueError(
                f"auto_turbulence_inlet '{marker}' has no prescribed Dirichlet "
                'velocity boundary condition.')

        # TODO: Support multiple inlet markers and form the uniform IC from
        # flow-rate-weighted inlet turbulence quantities.
        # TODO: Re-evaluate automatic inlet values for time-dependent velocity
        # data; the present implementation intentionally uses only t=0.
        inlet_velocity = velocity_markers[marker][0]
        boundary = self.mesh.Boundaries(marker)
        area = float(ngs.Integrate(1.0, self.mesh, definedon=boundary))
        if area <= 0.0:
            raise ValueError(
                f"auto_turbulence_inlet '{marker}' has zero boundary measure.")

        normal = ngs.specialcf.normal(self.mesh.dim)
        outward_flux = float(ngs.Integrate(
            inlet_velocity * normal, self.mesh, definedon=boundary))
        flux_tolerance = 1e-14 * max(1.0, area)
        if outward_flux >= -flux_tolerance:
            raise ValueError(
                f"auto_turbulence_inlet '{marker}' must have nonzero inward net "
                f'flux; computed outward flux is {outward_flux:.6g}.')

        bulk_velocity = -outward_flux / area
        reynolds = (bulk_velocity * self.turbulence_hydraulic_diameter
                    / self.kv[0])
        intensity = 0.16 * reynolds ** (-1.0 / 8.0)
        k_value = 1.5 * (bulk_velocity * intensity) ** 2
        length_scale = (self.turbulence_length_scale_ratio
                        * self.turbulence_hydraulic_diameter)
        epsilon_value = (self.C_mu ** 0.75 * k_value ** 1.5
                         / length_scale)

        dirichlet = self.BC.setdefault('dirichlet', {})
        for component, value in (('k', k_value), ('epsilon', epsilon_value)):
            component_boundaries = dirichlet.setdefault(component, {})
            if marker not in component_boundaries:
                component_boundaries[marker] = [value] * len(self.t_param)
                existing = self.dirichlet_names.get(component, '')
                names = [name for name in existing.split('|') if name]
                if marker not in names:
                    names.append(marker)
                self.dirichlet_names[component] = '|'.join(names)

        ic_dict = self.ic_functions.ic_dict.get(self.name(), {})
        explicit_all = 'all' in ic_dict
        if not explicit_all and 'k' not in ic_dict:
            self.IC.components[self.model_components_ic['k']].Set(k_value)
        if not explicit_all and 'epsilon' not in ic_dict:
            self.IC.components[self.model_components_ic['epsilon']].Set(
                epsilon_value)

    def _regularized_turbulence(self, time_step: int):
        """Floor-bounded k and epsilon, for use in ratios like k**2/epsilon.

        Guards against division by zero and negative coefficients during
        nonlinear iterations. Does not modify the stored ``UIter`` fields.
        """
        comp = self.model_components
        k_safe = Max(self._k_cell, ngs.CoefficientFunction(self.k_floor))
        epsilon_safe = Max(
            self._epsilon_cell, ngs.CoefficientFunction(self.epsilon_floor))
        # epsilon >= C_mu k**2 / (max_viscosity_ratio * nu), tied to local k so
        # C_mu k**2/epsilon stays under the ratio by construction. A scalar
        # limiter bound can't express this, so it's applied here instead.
        epsilon_safe = Max(
            epsilon_safe,
            self.C_mu * k_safe ** 2
            / (self.max_viscosity_ratio * self.kv[time_step]))
        return k_safe, epsilon_safe

    def _update_wall_function(self) -> None:
        """Refresh wall data from the current lagged turbulence and velocity."""
        comp = self.model_components
        self._wallf.update(
            self.UIter.components[comp['k']],
            self.UIter.components[comp[self.TURBULENCE_VELOCITY]])

    def _build_turbulent_viscosity(self, time_step: int):
        k, epsilon = self._regularized_turbulence(time_step)
        comp = self.model_components
        k_raw = self.UIter.components[comp['k']]
        epsilon_raw = self.UIter.components[comp['epsilon']]
        if self._bounded:
            # An epsilon at its floor is a clamped undershoot, not a physical
            # state; trusting k**2/epsilon there would inflate nu_t to the cap.
            # Fade it out smoothly instead (see _epsilon_trust).
            bulk_valid = self._epsilon_trust()
        else:
            bulk_valid = ngs.IfPos(
                k_raw, ngs.IfPos(epsilon_raw, 1.0, 0.0), 0.0)

        if self._wallf is not None:
            nu_t = self._wallf.eval_nu_t(k, epsilon)
            wall_mask = self._wallf.near_wall_mask()
            # Wall law stays active throughout the wall layer; in the bulk,
            # bulk_valid switches turbulence off instead of trusting the floor.
            nu_t = wall_mask * nu_t + (1.0 - wall_mask) * bulk_valid * nu_t
        else:
            nu_t = bulk_valid * self.C_mu * k ** 2 / epsilon

        if self.realizability_limiter:
            nu_t = self._realizable_nu_t(nu_t, k)

        cap = self.max_viscosity_ratio * self.kv[time_step]
        return ngs.IfPos(nu_t, ngs.IfPos(cap - nu_t, nu_t, cap), 0.0)

    def _strain_magnitude(self):
        """S = sqrt(2 S_ij S_ij) from the lagged velocity."""
        velocity = self.UIter.components[
            self.model_components[self.TURBULENCE_VELOCITY]]
        strain = 0.5 * (ngs.grad(velocity) + ngs.grad(velocity).trans)
        return ngs.sqrt(2.0 * ngs.InnerProduct(strain, strain) + 1e-30)

    def _realizable_nu_t(self, nu_t, k):
        """Durbin (1996) realizability bound on the turbulent time scale.

        T = min(k/epsilon, a / (sqrt(6) C_mu S)) with nu_t = C_mu k T, i.e.
        nu_t <= a k / (sqrt(6) S). Follows from positivity of the normal Reynolds
        stresses, so it caps nu_t by the local strain instead of letting k**2
        over a collapsing epsilon run to the viscosity ratio.
        """
        bound = (self.realizability_coefficient * k
                 / (np.sqrt(6.0) * self._strain_magnitude()))
        return ngs.IfPos(bound - nu_t, nu_t, bound)

    def _get_turbulent_viscosity(self, time_step: int):
        return self._turbulent_viscosity[time_step]

    def _get_effective_viscosity(self, time_step: int):
        return self.kv[time_step] + self._get_turbulent_viscosity(time_step)

    def _epsilon_trust(self):
        """Smooth 0..1 weight that vanishes as epsilon nears its floor (a clamped
        undershoot there, not a physical state). Healthy cells get weight ~1.
        ponytail: 100 is a trust margin, not physics.
        """
        epsilon_raw = self._epsilon_cell
        return epsilon_raw / (epsilon_raw + 100.0 * self.epsilon_floor)

    def _epsilon_k_ratio(self, k, epsilon):
        """Smooth, upper-bounded epsilon/k reaction coefficient.

        Raw epsilon/k diverges as k -> 0; this form tends to max_epsilon_k_ratio
        instead. The trust weight also removes the artificial O(1) sink that
        forms where both k and epsilon sit at their floors.
        """
        ratio = epsilon / (k + epsilon / self.max_epsilon_k_ratio)
        if self._bounded:
            ratio = ratio * self._epsilon_trust()
        return ratio

    def _limit_production(self, production, epsilon):
        """Apply the configured bulk production cap uniformly in every cell."""
        limit = self.production_limit_coefficient * epsilon
        return ngs.IfPos(limit - production, production, limit)

    def _production(self, time_step: int):
        velocity = self.UIter.components[
            self.model_components[self.TURBULENCE_VELOCITY]]
        nu_t = self._get_turbulent_viscosity(time_step)
        strain = 0.5 * (ngs.grad(velocity) + ngs.grad(velocity).trans)
        production = 2.0 * nu_t * ngs.InnerProduct(strain, strain)
        production = ngs.IfPos(production, production, 0.0)

        if self.production_limiter:
            _, epsilon = self._regularized_turbulence(time_step)
            production = self._limit_production(production, epsilon)

        return production

    def _get_turbulence_wind(self, gfu, time_step: int):
        """Return the velocity used to convect k and epsilon."""
        return self._get_wind(gfu, time_step)

    def _additional_turbulence_sources(self, time_step: int):
        """Model-specific additions to the k and epsilon source terms."""
        zero = ngs.CoefficientFunction(0.0)
        return zero, zero

    def _neumann_markers(self, component: str) -> str:
        markers = self.BC.get('neumann', {}).get(component, {}).keys()
        if component == 'epsilon' and self._wallf is not None:
            markers = (marker for marker in markers
                       if marker != self.wall_boundary)
        return '|'.join(markers)

    def _epsilon_dirichlet_markers(self) -> str:
        """Configured epsilon boundaries plus the active wall-function wall."""
        markers = list(self.BC.get('dirichlet', {}).get('epsilon', {}).keys())
        if self._wallf is not None and self.wall_boundary not in markers:
            markers.append(self.wall_boundary)
        return '|'.join(markers)

    def _add_scalar_transport(self, form, scalar, test, wind, diffusivity,
                              dirichlet_markers: str, neumann_markers: str,
                              dt: Parameter):
        n = self._normal
        form += -dt * scalar * (wind * ngs.grad(test)) * ngs.dx
        form += dt * diffusivity * ngs.grad(scalar) * ngs.grad(test) * ngs.dx

        if self.DG:
            wind_n = wind * n
            flux = avg(scalar) * wind_n + 0.5 * ngs.Norm(wind_n) * jump(scalar)
            form += dt * jump(test) * flux * ngs.dx(skeleton=True)
            facet_diffusivity = ngs.CoefficientFunction(diffusivity)
            avg_diffusivity = avg(facet_diffusivity)
            avg_diffusive_grad_test = weighted_grad_avg(
                test, facet_diffusivity)
            avg_diffusive_grad_scalar = weighted_grad_avg(
                scalar, facet_diffusivity)
            form += -dt * (n * avg_diffusive_grad_test) * jump(
                scalar) * ngs.dx(skeleton=True)
            form += dt * (
                avg_diffusivity * self._penalty * jump(scalar)
                - avg_diffusive_grad_scalar * n
            ) * jump(test) * ngs.dx(skeleton=True)

            if dirichlet_markers:
                form += dt * test * (
                    0.5 * scalar * wind_n + 0.5 * scalar * ngs.Norm(wind_n)
                ) * self._ds(dirichlet_markers)
                form += -dt * diffusivity * scalar * (
                    ngs.grad(test) * n) * self._ds(dirichlet_markers)
                form += dt * diffusivity * (
                    self._penalty * scalar - ngs.grad(scalar) * n
                ) * test * self._ds(dirichlet_markers)

            if neumann_markers:
                form += dt * test * scalar * Max(
                    wind_n, ngs.CoefficientFunction(0.0)
                ) * self._ds(neumann_markers)

        return form

    def construct_bilinear_time_ODE(
            self, U: Union[List[ProxyFunction], List[GridFunction]],
            V: List[ProxyFunction], dt: Parameter = Parameter(1.0),
            time_step: int = 0) -> List[BilinearForm]:
        forms = super().construct_bilinear_time_ODE(U, V, dt, time_step)
        comp = self.model_components
        wind = self._get_turbulence_wind(U, time_step)
        nu_t = self._get_turbulent_viscosity(time_step)
        k = U[comp['k']]
        epsilon = U[comp['epsilon']]
        zeta = V[comp['k']]
        psi = V[comp['epsilon']]

        d_k = self.kv[time_step] + nu_t / self.sigma_k
        d_epsilon = self.kv[time_step] + nu_t / self.sigma_epsilon
        k_previous, epsilon_previous = self._regularized_turbulence(time_step)
        form = forms[0]
        form = self._add_scalar_transport(
            form, k, zeta, wind, d_k,
            self.dirichlet_names.get('k', ''), self._neumann_markers('k'), dt)
        form = self._add_scalar_transport(
            form, epsilon, psi, wind, d_epsilon,
            self._epsilon_dirichlet_markers(),
            self._neumann_markers('epsilon'), dt)
        # Sink terms go on the implicit side with lagged coefficients only
        # (standard segregated k-epsilon linearization); on the RHS they make
        # the Picard iteration unstable when k or epsilon changes rapidly.
        epsilon_k_ratio = self._epsilon_k_ratio(k_previous, epsilon_previous)
        form += dt * epsilon_k_ratio * k * zeta * ngs.dx
        form += dt * self.C_2 * (
            epsilon_k_ratio) * epsilon * psi * ngs.dx
        forms[0] = form
        return forms

    def construct_linear(self, V: List[ProxyFunction],
                         gfu_0: Optional[List[GridFunction]], dt: Parameter,
                         time_step: int) -> List[LinearForm]:
        forms = super().construct_linear(V, gfu_0, dt, time_step)
        comp = self.model_components
        zeta = V[comp['k']]
        psi = V[comp['epsilon']]
        wind = self._get_turbulence_wind(gfu_0, time_step)
        k, epsilon = self._regularized_turbulence(time_step)
        nu_t = self._get_turbulent_viscosity(time_step)
        production = self._production(time_step)
        additional_k, additional_epsilon = \
            self._additional_turbulence_sources(time_step)
        form = forms[0]

        source_k = self.f.get('k', [0.0] * len(self.t_param))[time_step]
        source_epsilon = self.f.get(
            'epsilon', [0.0] * len(self.t_param))[time_step]
        form += dt * (
            production + source_k + additional_k
        ) * zeta * ngs.dx
        form += dt * (
            self.C_1 * self._epsilon_k_ratio(k, epsilon) * production
            + source_epsilon + additional_epsilon
        ) * psi * ngs.dx

        if self.DG:
            n = self._normal
            d_k = self.kv[time_step] + nu_t / self.sigma_k
            d_epsilon = (
                self.kv[time_step] + nu_t / self.sigma_epsilon)
            for component, test, diffusivity in (
                    ('k', zeta, d_k), ('epsilon', psi, d_epsilon)):
                for marker, values in self.BC.get(
                        'dirichlet', {}).get(component, {}).items():
                    value = values[time_step]
                    wind_n = wind * n
                    form += -dt * test * (
                        0.5 * value * wind_n
                        - 0.5 * value * ngs.Norm(wind_n)
                    ) * self._ds(marker)
                    form += dt * diffusivity * self._penalty * value * test * self._ds(marker)
                    form += -dt * diffusivity * value * (
                        ngs.grad(test) * n) * self._ds(marker)

            if self._wallf is not None:
                # Impose the wall-law epsilon through the same weak DG/Nitsche
                # terms as configured Dirichlet data, using Picard-lagged k.
                value = self._wallf.epsilon_wall_cell(k)
                wind_n = wind * n
                form += -dt * psi * (
                    0.5 * value * wind_n
                    - 0.5 * value * ngs.Norm(wind_n)
                ) * self._ds(self.wall_boundary)
                form += (dt * d_epsilon * self._penalty * value * psi
                         * self._ds(self.wall_boundary))
                form += -dt * d_epsilon * value * (
                    ngs.grad(psi) * n) * self._ds(self.wall_boundary)

        for component, test in (('k', zeta), ('epsilon', psi)):
            for marker, values in self.BC.get(
                    'neumann', {}).get(component, {}).items():
                if (component == 'epsilon' and self._wallf is not None
                        and marker == self.wall_boundary):
                    continue
                form += -dt * test * values[time_step] * self._ds(marker)

        forms[0] = form
        return forms

    def solve_single_step(self, a_lst: List[BilinearForm],
                          L_lst: List[LinearForm],
                          precond_lst: List[Preconditioner],
                          gfu: GridFunction, time_step: int = 0) -> None:
        comp = self.model_components
        if (gfu.components[comp['k']].vec.Norm() == 0.0
                or gfu.components[comp['epsilon']].vec.Norm() == 0.0):
            gfu.vec.data = self.IC.vec

        previous = ngs.GridFunction(self.fes)

        for iteration in range(self.nonlinear_max_iters):
            previous.vec.data = gfu.vec
            self.UIter.vec.data = gfu.vec
            self.W[0].vec.data = gfu.components[comp['u']].vec

            if self._wallf is not None:
                self._update_wall_function()

            self.apply_dirichlet_bcs_to(gfu, time_step)
            a_lst[0].Assemble()
            L_lst[0].Assemble()
            if precond_lst[0] is not None:
                precond_lst[0].Update()
            self.linear_solve(a_lst[0], L_lst[0], precond_lst[0], gfu)

            for index, factor in enumerate(self.relaxation_factors):
                if factor < 1.0:
                    self._relax_component_on_free_dofs(
                        gfu.components[index], previous.components[index],
                        factor, self.fes.components[index])

            self._bound_turbulence(gfu)

            difference = gfu.vec.CreateVector()
            difference.data = gfu.vec - previous.vec
            tolerance = (
                self.abs_nonlinear_tolerance
                + self.rel_nonlinear_tolerance * gfu.vec.Norm())
            if difference.Norm() < tolerance:
                logging.info(
                    'KEpsilonINS converged in %d nonlinear iteration(s).',
                    iteration + 1)
                break
        else:
            logging.warning(
                'KEpsilonINS did not converge within %d nonlinear iterations.',
                self.nonlinear_max_iters)

        self.UIter.vec.data = gfu.vec
        self.W[0].vec.data = gfu.components[comp['u']].vec

    def linearized_solve(self, a_assembled: BilinearForm, L_assembled: LinearForm,
                         precond: Preconditioner, gfu: GridFunction):
        """Perform one stationary Picard update."""
        previous = None
        if self.k_epsilon_diagnostics:
            previous = ngs.GridFunction(self.fes)
            previous.vec.data = self.UIter.vec

        super().linearized_solve(a_assembled, L_assembled, precond, gfu)

        for index, factor in enumerate(self.relaxation_factors):
            if factor < 1.0:
                self._relax_component_on_free_dofs(
                    gfu.components[index], self.UIter.components[index],
                    factor, self.fes.components[index])

        self._bound_turbulence(gfu)

        comp = self.model_components
        error_squared = 0.0
        norm_squared = 0.0
        for component in ('u', 'k', 'epsilon'):
            index = comp[component]
            difference = gfu.components[index].vec.CreateVector()
            difference.data = (
                gfu.components[index].vec - self.UIter.components[index].vec)
            error_squared += difference.Norm() ** 2
            norm_squared += gfu.components[index].vec.Norm() ** 2

        if self.k_epsilon_diagnostics:
            self._diagnostic_iteration += 1
            self._write_picard_diagnostics(previous, gfu,
                                           self._diagnostic_iteration)

        return error_squared ** 0.5, norm_squared ** 0.5

    def _diagnostic_project_closures(self, state: GridFunction):
        """Project closures for ``state`` without changing the Picard iterate."""
        saved = self.UIter.vec.Copy()
        saved.data = self.UIter.vec
        p0 = ngs.L2(self.mesh, order=0)
        nu_t = ngs.GridFunction(p0)
        production = ngs.GridFunction(p0)
        production_raw = ngs.GridFunction(p0)
        y_plus = ngs.GridFunction(p0)
        try:
            self.UIter.vec.data = state.vec
            if self._wallf is not None:
                self._update_wall_function()
                y_plus.vec.data = self._wallf.y_plus_cell().vec
            nu_t.Set(self._get_turbulent_viscosity(0))
            production.Set(self._production(0))
            velocity = self.UIter.components[
                self.model_components[self.TURBULENCE_VELOCITY]]
            strain = 0.5 * (ngs.grad(velocity) + ngs.grad(velocity).trans)
            production_raw.Set(
                2.0 * nu_t * ngs.InnerProduct(strain, strain))
        finally:
            self.UIter.vec.data = saved
            if self._wallf is not None:
                self._update_wall_function()
        return nu_t, production, production_raw, y_plus

    @staticmethod
    def _diagnostic_update_norm(current, previous):
        difference = current.vec.CreateVector()
        difference.data = current.vec - previous.vec
        absolute = difference.Norm()
        relative = absolute / max(current.vec.Norm(), 1e-30)
        return absolute, relative

    def _diagnostic_cell_location(self, dof: int):
        """Return the element number and centroid owning one P0 DOF."""
        p0 = self._wallf._fes0 if self._wallf is not None else ngs.L2(
            self.mesh, order=0)
        for element in p0.Elements(ngs.VOL):
            if dof not in element.dofs:
                continue
            mesh_element = self.mesh[ngs.ElementId(ngs.VOL, element.nr)]
            points = [np.asarray(self.mesh.vertices[vertex.nr].point[:self.mesh.dim],
                                 dtype=float)
                      for vertex in mesh_element.vertices]
            centroid = np.mean(points, axis=0)
            coordinates = list(centroid) + [float('nan')] * (3 - len(centroid))
            return element.nr, coordinates[0], coordinates[1], coordinates[2]
        return -1, float('nan'), float('nan'), float('nan')

    def _diagnostic_wall_velocity(self, state: GridFunction):
        """Measure physical wall trace error and wall-cell ``u_y`` oscillation."""
        if self._wallf is None:
            return {
                'wall_normal_trace_max': float('nan'),
                'wall_cell_uy_abs_max': float('nan'),
                'opposite_uy_wall_pairs': 0,
                'max_adjacent_uy_jump': float('nan'),
                'max_uy_jump_element_a': -1,
                'max_uy_jump_element_b': -1,
            }

        p0 = self._wallf._fes0
        velocity = state.components[
            self.model_components[self.TURBULENCE_VELOCITY]]
        test = p0.TestFunction()
        normal = ngs.specialcf.normal(self.mesh.dim)
        trace_form = ngs.LinearForm(p0)
        trace_form += test * ngs.sqrt((velocity * normal) ** 2) * ngs.ds(
            definedon=self.mesh.Boundaries(self.wall_boundary), skeleton=True)
        trace_form.Assemble()
        trace_integrals = np.asarray(trace_form.vec).ravel()
        owners = self._wallf._marked
        trace_average = np.zeros(p0.ndof)
        trace_average[owners] = (
            trace_integrals[owners] / self._wallf._wall_measure[owners])

        uy = ngs.GridFunction(p0)
        uy.Set(velocity[1])
        uy_values = uy.vec.FV().NumPy()
        wall = self._wallf.near_wall_mask().vec.FV().NumPy() > 0.5

        facet_to_elements = {}
        element_dof = {}
        for element in p0.Elements(ngs.VOL):
            element_dof[element.nr] = element.dofs[0]
            for facet in element.facets:
                facet_to_elements.setdefault(facet.nr, []).append(element.nr)

        opposite_pairs = 0
        maximum_jump = -1.0
        maximum_pair = (-1, -1)
        for adjacent in facet_to_elements.values():
            if len(adjacent) != 2:
                continue
            first, second = adjacent
            first_dof, second_dof = element_dof[first], element_dof[second]
            if not (wall[first_dof] and wall[second_dof]):
                continue
            first_uy, second_uy = uy_values[first_dof], uy_values[second_dof]
            if first_uy * second_uy < 0.0:
                opposite_pairs += 1
            jump = abs(first_uy - second_uy)
            if jump > maximum_jump:
                maximum_jump = jump
                maximum_pair = (first, second)

        return {
            'wall_normal_trace_max': float(np.max(trace_average[owners])),
            'wall_cell_uy_abs_max': float(np.max(np.abs(uy_values[wall]))),
            'opposite_uy_wall_pairs': opposite_pairs,
            'max_adjacent_uy_jump': maximum_jump,
            'max_uy_jump_element_a': maximum_pair[0],
            'max_uy_jump_element_b': maximum_pair[1],
        }

    def _write_picard_diagnostics(self, previous: GridFunction,
                                  current: GridFunction, iteration: int) -> None:
        """Write scalar diagnostics and optional VTU fields for one Picard step."""
        comp = self.model_components
        update_values = {}
        for name in ('u', 'k', 'epsilon'):
            update_values[name] = self._diagnostic_update_norm(
                current.components[comp[name]], previous.components[comp[name]])

        old_nut, old_pk, _, old_yplus = self._diagnostic_project_closures(previous)
        new_nut, new_pk, raw_pk, new_yplus = self._diagnostic_project_closures(current)
        dnut_abs, dnut_rel = self._diagnostic_update_norm(new_nut, old_nut)
        dpk_abs, dpk_rel = self._diagnostic_update_norm(new_pk, old_pk)

        y_old = old_yplus.vec.FV().NumPy()
        y_new = new_yplus.vec.FV().NumPy()
        if self._wallf is not None:
            wall = self._wallf.near_wall_mask().vec.FV().NumPy() > 0.5
        else:
            wall = np.zeros_like(y_new, dtype=bool)
        wall_y = y_new[wall]
        viscous_to_log = int(np.count_nonzero(
            wall & (y_old <= self._wallf.YPLUS_VISCOUS)
            & (y_new > self._wallf.YPLUS_VISCOUS))) if self._wallf else 0
        log_to_viscous = int(np.count_nonzero(
            wall & (y_old > self._wallf.YPLUS_VISCOUS)
            & (y_new <= self._wallf.YPLUS_VISCOUS))) if self._wallf else 0

        p0 = new_nut.space
        k_cell = ngs.GridFunction(p0)
        epsilon_cell = ngs.GridFunction(p0)
        k_cell.Set(current.components[comp['k']])
        epsilon_cell.Set(current.components[comp['epsilon']])
        k_values = k_cell.vec.FV().NumPy()
        epsilon_values = epsilon_cell.vec.FV().NumPy()
        nut_values = new_nut.vec.FV().NumPy()
        raw_pk_values = raw_pk.vec.FV().NumPy()
        production_limit = self.production_limit_coefficient * np.maximum(
            epsilon_values, self.epsilon_floor)

        dnut = np.abs(new_nut.vec.FV().NumPy()
                      - old_nut.vec.FV().NumPy())
        worst_dof = int(np.argmax(dnut)) if dnut.size else -1
        worst_element, worst_x, worst_y, worst_z = self._diagnostic_cell_location(
            worst_dof)
        wall_velocity = self._diagnostic_wall_velocity(current)

        row = {
            'iteration': iteration,
            'du_abs': update_values['u'][0], 'du_rel': update_values['u'][1],
            'dk_abs': update_values['k'][0], 'dk_rel': update_values['k'][1],
            'deps_abs': update_values['epsilon'][0],
            'deps_rel': update_values['epsilon'][1],
            'dnut_abs': dnut_abs, 'dnut_rel': dnut_rel,
            'dPk_abs': dpk_abs, 'dPk_rel': dpk_rel,
            'yplus_min': float(np.min(wall_y)) if wall_y.size else float('nan'),
            'yplus_median': float(np.median(wall_y)) if wall_y.size else float('nan'),
            'yplus_max': float(np.max(wall_y)) if wall_y.size else float('nan'),
            'cells_yplus_below_11_25': int(np.count_nonzero(
                wall_y <= self._wallf.YPLUS_VISCOUS)) if self._wallf else 0,
            'cells_viscous_to_log': viscous_to_log,
            'cells_log_to_viscous': log_to_viscous,
            'cells_yplus_below_30': int(np.count_nonzero(wall_y < 30.0)),
            'cells_yplus_above_300': int(np.count_nonzero(wall_y > 300.0)),
            'nut_wall_min': float(np.min(nut_values[wall])) if wall_y.size else float('nan'),
            'nut_wall_max': float(np.max(nut_values[wall])) if wall_y.size else float('nan'),
            'nut_bulk_max': float(np.max(nut_values[~wall])) if np.any(~wall) else float('nan'),
            'k_floor_cells': int(np.count_nonzero(
                k_values <= self.k_floor * (1.0 + 1e-8))),
            'epsilon_floor_cells': int(np.count_nonzero(
                epsilon_values <= self.epsilon_floor * (1.0 + 1e-8))),
            'production_limiter_cells': int(np.count_nonzero(
                raw_pk_values > production_limit * (1.0 + 1e-10)))
                if self.production_limiter else 0,
            'max_dnut': float(dnut[worst_dof]) if worst_dof >= 0 else float('nan'),
            'max_dnut_element': worst_element,
            'max_dnut_x': worst_x, 'max_dnut_y': worst_y,
            'max_dnut_z': worst_z,
        }
        limiter_defaults = {
            'modified_elements': 0,
            'mean_clamped_elements': 0,
            'higher_mode_scaled_elements': 0,
            'below_floor_cell_means': 0,
            'negative_cell_means': 0,
            'raw_cell_mean_min': float('nan'),
            'raw_bezier_ordinate_min': float('nan'),
            'correction_norm': 0.0,
        }
        latest_limiter = getattr(self, '_latest_limiter_diagnostics', {})
        for component in ('k', 'epsilon'):
            stats = dict(limiter_defaults)
            stats.update(latest_limiter.get(component, {}))
            for name, value in stats.items():
                row[f'{component}_limiter_{name}'] = value
        row.update(wall_velocity)

        mode = 'a' if self._diagnostic_csv_initialized else 'w'
        with self._diagnostic_csv.open(mode, newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not self._diagnostic_csv_initialized:
                writer.writeheader()
            writer.writerow(row)
        self._diagnostic_csv_initialized = True

        logging.info(
            'k-epsilon diagnostic %d: du=%.3e dk=%.3e deps=%.3e '
            'dnut=%.3e dPk=%.3e branch_crossings[up/down]=%d/%d '
            'floors[k/epsilon]=%d/%d production_cap=%d max_dnut_cell=%d '
            'wall_un_max=%.3e opposite_uy_pairs=%d '
            'limited[k/epsilon]=%d/%d mean_clamped=%d/%d correction=%.3e/%.3e',
            iteration, update_values['u'][0], update_values['k'][0],
            update_values['epsilon'][0], dnut_rel, dpk_rel,
            viscous_to_log, log_to_viscous, row['k_floor_cells'],
            row['epsilon_floor_cells'], row['production_limiter_cells'],
            worst_element, row['wall_normal_trace_max'],
            row['opposite_uy_wall_pairs'],
            row['k_limiter_modified_elements'],
            row['epsilon_limiter_modified_elements'],
            row['k_limiter_mean_clamped_elements'],
            row['epsilon_limiter_mean_clamped_elements'],
            row['k_limiter_correction_norm'],
            row['epsilon_limiter_correction_norm'])

        if (self.k_epsilon_diagnostic_vtu_start <= iteration
                <= self.k_epsilon_diagnostic_vtu_end):
            velocity_delta = (current.components[comp['u']]
                              - previous.components[comp['u']])
            k_delta = (current.components[comp['k']]
                       - previous.components[comp['k']])
            epsilon_delta = (current.components[comp['epsilon']]
                             - previous.components[comp['epsilon']])
            ngs.VTKOutput(
                ma=self.mesh,
                coefs=[velocity_delta, k_delta, epsilon_delta,
                       new_nut - old_nut, new_nut, new_pk, new_yplus],
                names=['delta_u', 'delta_k', 'delta_epsilon', 'delta_nu_t',
                       'nu_t', 'production_k', 'y_plus'],
                filename=str(self._diagnostic_dir / f'picard_{iteration:03d}'),
                subdivision=1,
            ).Do()

    def update_linearization(self, gfu: GridFunction) -> None:
        self._bound_turbulence(gfu)
        super().update_linearization(gfu)
        self.UIter.vec.data = gfu.vec
        if self._wallf is not None:
            self._update_wall_function()


class KEpsilonINS(KEpsilonModel, INS):
    """Single-phase incompressible RANS model with a k-epsilon closure."""


class KEpsilonTFM(KEpsilonModel, TwoFluidModel):
    """Euler--Euler two-fluid model with continuous-phase k-epsilon turbulence."""

    TURBULENCE_VELOCITY = 'u_c'

    def _get_turbulence_wind(self, gfu, time_step: int):
        # TFM is Picard-linearized: both transport equations use the lagged
        # continuous-phase velocity, never the dispersed-phase velocity.
        return self.UIter.components[self.model_components['u_c']]

    def _phase_viscosity(self, phase: str, time_step: int):
        molecular = super()._phase_viscosity(phase, time_step)
        if phase == 'c':
            return molecular + self._get_turbulent_viscosity(time_step)
        return molecular

    def _additional_turbulence_sources(self, time_step: int):
        """Dispersed-phase sources S_k and S_epsilon."""
        if not self.drag_switch:
            return super()._additional_turbulence_sources(time_step)

        comp = self.model_components
        wc = self.UIter.components[comp['u_c']]
        wd = self.UIter.components[comp['u_d']]
        alpha_c = self.UIter.components[comp['alpha_c']]
        alpha_d = 1.0 - alpha_c
        k, epsilon = self._regularized_turbulence(time_step)

        rho_c = self.rho_c[time_step]
        rho_d = self.rho_d[time_step]
        nu_c = self.nu_c[time_step]
        diameter = self.dp[time_step]
        surface_tension = self.sigma_c[time_step]
        nu_t = self._get_turbulent_viscosity(time_step)
        drag = self._get_drag_coeff(wd, wc, alpha_d, time_step)
        drag_multiplier = (
            0.75 * drag * alpha_d * rho_c / diameter * ngs.Norm(wd - wc)
        )

        length_scale = self.C_mu * k ** 1.5 / epsilon
        turbulent_reynolds = (
            length_scale * ngs.sqrt(2.0 / 3.0 * k) / nu_c
        )
        beta = (
            2.0 * drag_multiplier * length_scale ** 2
            / (rho_c * nu_c * turbulent_reynolds + 1e-30)
        )
        c_t = (3.0 + beta) / (1.0 + beta + 2.0 * rho_d / rho_c)

        source_k = (
            2.0 * k * alpha_d * drag_multiplier * (c_t - 1.0) / rho_c
            + drag_multiplier * nu_t / (rho_c * surface_tension)
            * (-ngs.grad(alpha_c) * (wc - wd)) / (alpha_c + 1e-30)
        )
        source_epsilon = (
            2.0 * epsilon * alpha_d * drag_multiplier * (c_t - 1.0)
            / rho_c
        )
        return source_k, source_epsilon

    def _cold_start_required(self, gfu: GridFunction) -> bool:
        comp = self.model_components
        return (super()._cold_start_required(gfu)
                or gfu.components[comp['k']].vec.Norm() == 0.0
                or gfu.components[comp['epsilon']].vec.Norm() == 0.0)

    def _prepare_picard_iteration(self) -> None:
        if self._wallf is not None:
            self._update_wall_function()

    def _postprocess_picard_solution(self, gfu: GridFunction) -> None:
        self._bound_turbulence(gfu)

    def solve_single_step(self, a_lst: List[BilinearForm],
                          L_lst: List[LinearForm],
                          precond_lst: List[Preconditioner],
                          gfu: GridFunction, time_step: int = 0) -> bool:
        return TwoFluidModel.solve_single_step(
            self, a_lst, L_lst, precond_lst, gfu, time_step)

    def linearized_solve(self, a_assembled: BilinearForm,
                         L_assembled: LinearForm, precond: Preconditioner,
                         gfu: GridFunction):
        error, solution_norm = TwoFluidModel.linearized_solve(
            self, a_assembled, L_assembled, precond, gfu)
        self._bound_turbulence(gfu)
        return error, solution_norm

    def update_linearization(self, gfu: GridFunction) -> None:
        self._bound_turbulence(gfu)
        TwoFluidModel.update_linearization(self, gfu)
        if self._wallf is not None:
            self._update_wall_function()
