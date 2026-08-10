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

"""Single-phase incompressible Navier--Stokes with a k-epsilon closure."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import ngsolve as ngs
from ngsolve import (BilinearForm, FESpace, GridFunction, LinearForm,
                     Parameter, Preconditioner)
from ngsolve.comp import ProxyFunction

from .ins import INS
from ..helpers.dg import avg, jump, weighted_grad_avg
from ..helpers.limiter import Limiter
from ..helpers.math import Max
from ..helpers.ngsolve_ import get_special_functions
from ..helpers.wall_func import KEpsilonWallFunction


class KEpsilonINS(INS):
    """RANS INS model closed with the standard high-Reynolds-number k-epsilon equations.

    Adds transport equations for turbulent kinetic energy and dissipation to INS.
    """

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
    }

    #: Optional ``[OTHER]`` switches
    DEFAULT_OPTIONS = {
        'wall_function': True,
        'wall_boundary': 'wall',
        'production_limiter': True,
        'realizability_limiter': True,
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
        for key in self.DEFAULT_OPTIONS:
            setattr(self, key, self._optional_config(key))

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        return {'u': 0, 'p': 1, 'k': 2, 'epsilon': 3}

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        return {'u': True, 'p': False, 'k': True, 'epsilon': True}

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        return [{'u': True, 'p': False, 'k': True, 'epsilon': True}]

    def _define_bc_types(self) -> List[str]:
        return super()._define_bc_types() + ['neumann']

    def _construct_fes(self) -> FESpace:
        spaces = self._construct_fes_helper()
        scalar_order = max(self.interp_ord - 1, 0)

        for component in ('k', 'epsilon'):
            element = self.element[component]
            kwargs = {
                'mesh': self.mesh,
                'order': scalar_order,
                'dgjumps': self.DG,
            }
            if element != 'L2':
                kwargs['dirichlet'] = self.dirichlet_names.get(component, '')
            spaces.append(getattr(ngs, element)(**kwargs))

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
        for component, floor in (('k', self.k_floor),
                                 ('epsilon', self.epsilon_floor)):
            field = gfu.components[comp[component]]
            # Pass the stable FES, not the per-call component, so the limiter's
            # cache keys match across iterations.
            space = self.fes.components[comp[component]]
            self._limiter.bezier_bound(field, space, space.globalorder,
                                       (floor, 1e20))

    def _set_model_parameters(self) -> None:
        super()._set_model_parameters()
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

        if self.production_limit_coefficient <= 0.0:
            raise ValueError('production_limit_coefficient must be positive.')
        if self.max_epsilon_k_ratio <= 0.0:
            raise ValueError('max_epsilon_k_ratio must be positive.')
        if self.wall_u_tau_method not in (0, 1):
            raise ValueError(
                'wall_u_tau_method must be 0 (k-based) or 1 (velocity-based).')

    def _post_init(self) -> None:
        super()._post_init()
        if self.linearize != 'Oseen':
            raise NotImplementedError('KEpsilonINS currently supports Oseen linearization only.')

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
            self.UIter.components[comp['u']])

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
        velocity = self.UIter.components[self.model_components['u']]
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
        velocity = self.UIter.components[self.model_components['u']]
        nu_t = self._get_turbulent_viscosity(time_step)
        strain = 0.5 * (ngs.grad(velocity) + ngs.grad(velocity).trans)
        production = 2.0 * nu_t * ngs.InnerProduct(strain, strain)
        production = ngs.IfPos(production, production, 0.0)

        if self.production_limiter:
            _, epsilon = self._regularized_turbulence(time_step)
            production = self._limit_production(production, epsilon)

        return production

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
        wind = self._get_wind(U, time_step)
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
        wind = self._get_wind(gfu_0, time_step)
        k, epsilon = self._regularized_turbulence(time_step)
        nu_t = self._get_turbulent_viscosity(time_step)
        production = self._production(time_step)
        form = forms[0]

        source_k = self.f.get('k', [0.0] * len(self.t_param))[time_step]
        source_epsilon = self.f.get(
            'epsilon', [0.0] * len(self.t_param))[time_step]
        form += dt * (
            production + source_k
        ) * zeta * ngs.dx
        form += dt * (
            self.C_1 * self._epsilon_k_ratio(k, epsilon) * production
            + source_epsilon
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
                    gfu.components[index].vec.data = (
                        factor * gfu.components[index].vec
                    + (1.0 - factor) * previous.components[index].vec)

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
        super().linearized_solve(a_assembled, L_assembled, precond, gfu)

        for index, factor in enumerate(self.relaxation_factors):
            if factor < 1.0:
                gfu.components[index].vec.data = (
                    factor * gfu.components[index].vec
                    + (1.0 - factor) * self.UIter.components[index].vec)

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

        return error_squared ** 0.5, norm_squared ** 0.5

    def update_linearization(self, gfu: GridFunction) -> None:
        self._bound_turbulence(gfu)
        super().update_linearization(gfu)
        self.UIter.vec.data = gfu.vec
        if self._wallf is not None:
            self._update_wall_function()
