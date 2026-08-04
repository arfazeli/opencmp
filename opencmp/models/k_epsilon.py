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

import ngsolve as ngs
from ngsolve import (BilinearForm, FESpace, GridFunction, LinearForm,
                     Parameter, Preconditioner)
from ngsolve.comp import ProxyFunction

from .ins import INS
from ..helpers.dg import avg, grad_avg, jump
from ..helpers.limiter import Limiter
from ..helpers.math import Max
from ..helpers.ngsolve_ import get_special_functions
from ..helpers.wall_func import KEpsilonWallFunction


class KEpsilonINS(INS):
    """RANS INS model closed with the standard high-Reynolds-number k-epsilon equations.

    The momentum and continuity equations are inherited from :class:`INS`.
    This class adds transport equations for turbulent kinetic energy and dissipation.
    """

    def _optional_config(self, key: str, value_type, default):
        try:
            value = self.config.get_item(['OTHER', key], value_type, quiet=True)
        except Exception:
            return default
        return default if value is None else value

    def _pre_init(self) -> None:
        self.wall_function = self._optional_config('wall_function', bool, False)
        self.wall_boundary = self._optional_config('wall_boundary', str, 'wall')
        self.slope_limiter = self._optional_config('slope_limiter', bool, False)
        self.production_limiter = self._optional_config(
            'production_limiter', bool, True)
        self.epsilon_wall_relaxation = self._optional_config(
            'epsilon_wall_relaxation', float, 0.15)
        self.epsilon_wall_max_change_factor = self._optional_config(
            'epsilon_wall_max_change_factor', float, 2.0)
        if not 0.0 < self.epsilon_wall_relaxation <= 1.0:
            raise ValueError('epsilon_wall_relaxation must be in (0, 1].')
        if self.epsilon_wall_max_change_factor < 1.0:
            raise ValueError('epsilon_wall_max_change_factor must be at least 1.')

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
                'order': scalar_order if element == 'L2' else self.interp_ord,
                'dgjumps': self.DG,
            }
            if element != 'L2':
                kwargs['dirichlet'] = self.dirichlet_names.get(component, '')
            spaces.append(getattr(ngs, element)(**kwargs))

        return FESpace(spaces, dgjumps=self.DG)

    def _set_model_parameters(self) -> None:
        super()._set_model_parameters()
        parameters = self.model_functions.model_parameters_dict
        self.C_mu = parameters['c_mu']['all']
        self.C_1 = parameters['c_1']['all']
        self.C_2 = parameters['c_2']['all']
        self.sigma_k = parameters['sigma_k']['all']
        self.sigma_epsilon = parameters['sigma_epsilon']['all']

        # Floors and the viscosity ratio cap are numerical safeguards, not
        # replacements for physically meaningful ICs and BCs.
        self.k_floor = parameters.get('k_floor', {'all': [1e-10] * len(self.t_param)})['all']
        self.epsilon_floor = parameters.get(
            'epsilon_floor', {'all': [1e-10] * len(self.t_param)})['all']
        self.max_viscosity_ratio = parameters.get(
            'max_viscosity_ratio', {'all': [1e5] * len(self.t_param)})['all']
        self.production_limit_coefficient = parameters.get(
            'production_limit_coefficient',
            {'all': [10.0] * len(self.t_param)})['all']

        if any(value <= 0.0 for value in self.production_limit_coefficient):
            raise ValueError('production_limit_coefficient must be positive.')

        if self.wall_function:
            self.density = parameters['density']['all']
            self.kappa = parameters['kappa']['all']
            self.E_log = parameters['e_log']['all']

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
        self._limiter = Limiter(self.mesh) if self.slope_limiter else None
        self._wallf = None
        self._epsilon_wall_element_dofs = {}
        self._epsilon_wall_values = None
        if self.wall_function:
            nu = self.kv[0]
            self._wallf = KEpsilonWallFunction(
                self.mesh,
                mu=self.density[0] * nu,
                rho=self.density[0],
                nu=nu,
                C_mu=self.C_mu[0],
                kappa=self.kappa[0],
                E=self.E_log[0],
                wall_boundary=self.wall_boundary,
            )
            self._wallf.update(
                self.UIter.components[self.model_components['u']])

            self._initialize_first_cell_epsilon()

        # Build and compile the turbulent viscosity once per time level so every
        # form reuses the same optimized evaluation tree.  This benefits both the
        # bulk k-epsilon expression and the deeper wall-law expression.
        self._turbulent_viscosity = []
        for time_step in range(len(self.t_param)):
            nu_t = self._build_turbulent_viscosity(time_step)
            self._turbulent_viscosity.append(nu_t.Compile())

        self._normal, _, self._penalty, _ = get_special_functions(self.mesh, self.nu)

    def _regularized_turbulence(self, time_step: int):
        """Return turbulence fields bounded below by their configured floors.

        The bounded fields are used in ratios such as ``k**2 / epsilon`` and
        ``epsilon / k`` to prevent division by zero and invalid negative
        coefficients during nonlinear iterations. The solution fields stored
        in ``UIter`` are not modified.

        Args:
            time_step: Index selecting the floor values for the current time level.

        Returns:
            The floor-bounded turbulent kinetic energy and dissipation fields.
        """
        comp = self.model_components
        k = self.UIter.components[comp['k']]
        epsilon = self.UIter.components[comp['epsilon']]
        k_safe = Max(k, ngs.CoefficientFunction(self.k_floor[time_step]))
        epsilon_safe = Max(
            epsilon, ngs.CoefficientFunction(self.epsilon_floor[time_step]))
        return k_safe, epsilon_safe

    def _build_turbulent_viscosity(self, time_step: int):
        k, epsilon = self._regularized_turbulence(time_step)
        comp = self.model_components
        k_raw = self.UIter.components[comp['k']]
        epsilon_raw = self.UIter.components[comp['epsilon']]
        velocity = self.UIter.components[self.model_components['u']]
        bulk_valid = ngs.IfPos(
            k_raw, ngs.IfPos(epsilon_raw, 1.0, 0.0), 0.0)

        if self._wallf is not None:
            nu_t = self._wallf.eval_nu_t(k, epsilon, velocity)
            wall_mask = self._wallf.near_wall_mask()
            # The wall law remains active throughout the topological wall layer.
            # In the bulk, an invalid raw turbulence state must switch turbulence
            # off instead of turning the epsilon floor into a huge viscosity.
            nu_t = wall_mask * nu_t + (1.0 - wall_mask) * bulk_valid * nu_t
        else:
            nu_t = bulk_valid * self.C_mu[time_step] * k ** 2 / epsilon

        cap = self.max_viscosity_ratio[time_step] * self.kv[time_step]
        return ngs.IfPos(nu_t, ngs.IfPos(cap - nu_t, nu_t, cap), 0.0)

    def _get_turbulent_viscosity(self, time_step: int):
        return self._turbulent_viscosity[time_step]

    def _get_kinematic_viscosity(self, time_step: int):
        return self.kv[time_step] + self._get_turbulent_viscosity(time_step)

    def _initialize_first_cell_epsilon(self) -> None:
        """Initialize the algebraic epsilon treatment in wall-adjacent cells.

        This maps cells selected by the wall-function topology mask to the
        corresponding epsilon-space DOFs. It then evaluates the equilibrium
        wall-law epsilon from the initial turbulent kinetic energy, stores one
        continuation value per wall cell, and applies those values to ``UIter``
        before the first Picard system is assembled.

        Raises:
            NotImplementedError: If epsilon does not use a discontinuous L2 space.
            RuntimeError: If a marked wall cell cannot be mapped to epsilon DOFs.
        """
        if not self.DG or self.element['epsilon'] != 'L2':
            raise NotImplementedError(
                'The first-cell epsilon treatment requires a discontinuous L2 '
                'epsilon space.')

        mask_values = self._wallf.first_cell_mask().vec.FV().NumPy()
        wall_elements = {
            element.nr
            for element in self._wallf._fes0.Elements(ngs.VOL)
            if any(mask_values[dof] > 0.5 for dof in element.dofs)
        }
        epsilon_space = self.fes.components[self.model_components['epsilon']]
        for element in epsilon_space.Elements(ngs.VOL):
            if element.nr not in wall_elements:
                continue
            self._epsilon_wall_element_dofs[element.nr] = tuple(element.dofs)

        if len(self._epsilon_wall_element_dofs) != len(wall_elements):
            raise RuntimeError('Could not map every wall-adjacent cell to epsilon DOFs.')

        # Start the nonlinear solve on the same equilibrium wall relation that
        # will be enforced later.  Initializing this continuation state from the
        # domain/inlet epsilon made every wall cell climb geometrically toward a
        # much larger target, so its absolute Picard update grew by construction.
        initial_k = ngs.GridFunction(self._wallf._fes0)
        initial_k.Set(self.UIter.components[self.model_components['k']])
        initial_target = ngs.GridFunction(self._wallf._fes0)
        initial_target.Set(self._wallf.epsilon_wall_cell(initial_k))
        initial_values = initial_target.vec.FV().NumPy()
        self._epsilon_wall_values = {
            element.nr: max(float(initial_values[element.dofs[0]]),
                            self.epsilon_floor[0])
            for element in self._wallf._fes0.Elements(ngs.VOL)
            if element.nr in wall_elements
        }
        # The coefficient functions assembled for the first Picard step read
        # UIter, so put the equilibrium values into that iterate immediately as
        # well as into the continuation history above.
        self._apply_first_cell_epsilon(self.UIter, 0)

    def _apply_first_cell_epsilon(self, gfu: GridFunction, time_step: int) -> None:
        """Update epsilon in wall-adjacent cells from the equilibrium wall law.

        The wall-law target is evaluated using the Picard-lagged turbulent
        kinetic energy in ``UIter``. Each cell value is limited relative to its
        previous value, under-relaxed, bounded by the epsilon floor, and then
        projected into every local epsilon DOF of the wall-adjacent element.
        Non-wall cells in ``gfu`` are left unchanged.

        Args:
            gfu: Solution grid function whose epsilon component is updated.
            time_step: Index selecting the epsilon floor for the current time level.
        """
        if self._wallf is None:
            return

        comp = self.model_components
        epsilon_component = gfu.components[comp['epsilon']]
        # Picard-lag the nonlinear wall target.  Using the just-solved k here
        # couples two new fields outside the assembled linear system and turns
        # the post-solve overwrite into an unrelaxed same-iteration feedback.
        k_cell = ngs.GridFunction(self._wallf._fes0)
        k_cell.Set(self.UIter.components[comp['k']])
        raw_target = ngs.GridFunction(self._wallf._fes0)
        raw_target.Set(self._wallf.epsilon_wall_cell(k_cell))
        raw_values = raw_target.vec.FV().NumPy()

        applied_cell = ngs.GridFunction(self._wallf._fes0)
        applied_cell_values = applied_cell.vec.FV().NumPy()
        mask_values = self._wallf.first_cell_mask().vec.FV().NumPy()
        epsilon_values = epsilon_component.vec.FV().NumPy()

        relaxation = self.epsilon_wall_relaxation
        change_factor = self.epsilon_wall_max_change_factor
        for element in self._wallf._fes0.Elements(ngs.VOL):
            if not any(mask_values[dof] > 0.5 for dof in element.dofs):
                continue
            cell_dof = element.dofs[0]
            previous = self._epsilon_wall_values[element.nr]
            target_value = max(float(raw_values[cell_dof]),
                               self.epsilon_floor[time_step])
            value, _ = self._relax_epsilon_wall_value(
                target_value, previous, relaxation, change_factor)
            value = max(value, self.epsilon_floor[time_step])
            applied_cell_values[cell_dof] = value
            self._epsilon_wall_values[element.nr] = value

        # Let NGSolve represent the P0 field in the actual high-order basis,
        # then copy every local coefficient on wall cells. This removes all
        # higher epsilon modes there without changing the global FESpace.
        projected = ngs.GridFunction(epsilon_component.space)
        projected.Set(applied_cell)
        projected_values = projected.vec.FV().NumPy()
        for dofs in self._epsilon_wall_element_dofs.values():
            epsilon_values[list(dofs)] = projected_values[list(dofs)]

    @staticmethod
    def _relax_epsilon_wall_value(raw_target: float, previous: float,
                                  relaxation: float,
                                  change_factor: float) -> tuple:
        """Limit and relax one algebraic first-cell epsilon target."""
        limited_target = min(
            max(raw_target, previous / change_factor),
            previous * change_factor)
        value = ((1.0 - relaxation) * previous
                 + relaxation * limited_target)
        return value, limited_target != raw_target

    def _limit_production(self, production, epsilon, time_step: int):
        """Use equilibrium limiting at walls and the configured limit in bulk."""
        bulk_limit = self.production_limit_coefficient[time_step] * epsilon
        bulk_production = ngs.IfPos(
            bulk_limit - production, production, bulk_limit)
        if self._wallf is None:
            return bulk_production

        # The high-Re wall treatment assumes local equilibrium, P_k ~= epsilon.
        # Apply that coefficient throughout the same expanded wall layer used by
        # wall viscosity and first-cell epsilon; retain the user's coefficient
        # only in the non-equilibrium interior.
        wall_production = ngs.IfPos(
            epsilon - production, production, epsilon)
        wall_mask = self._wallf.near_wall_mask()
        return (wall_mask * wall_production
                + (1.0 - wall_mask) * bulk_production)

    def _production(self, time_step: int):
        velocity = self.UIter.components[self.model_components['u']]
        nu_t = self._get_turbulent_viscosity(time_step)
        strain = 0.5 * (ngs.grad(velocity) + ngs.grad(velocity).trans)
        production = 2.0 * nu_t * ngs.InnerProduct(strain, strain)
        production = ngs.IfPos(production, production, 0.0)

        if self.production_limiter:
            _, epsilon = self._regularized_turbulence(time_step)
            production = self._limit_production(
                production, epsilon, time_step)

        return production

    def _neumann_markers(self, component: str) -> str:
        return '|'.join(self.BC.get('neumann', {}).get(component, {}).keys())

    def _epsilon_dirichlet_markers(self) -> str:
        """Configured epsilon boundaries; wall-function walls use zero flux."""
        return '|'.join(self.BC.get('dirichlet', {}).get('epsilon', {}).keys())

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
            form += -dt * diffusivity * (n * grad_avg(test)) * jump(scalar) * ngs.dx(skeleton=True)
            form += dt * diffusivity * (
                self._penalty * jump(scalar) - grad_avg(scalar) * n
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

        d_k = self.kv[time_step] + nu_t / self.sigma_k[time_step]
        d_epsilon = self.kv[time_step] + nu_t / self.sigma_epsilon[time_step]
        k_previous, epsilon_previous = self._regularized_turbulence(time_step)
        form = forms[0]
        form = self._add_scalar_transport(
            form, k, zeta, wind, d_k,
            self.dirichlet_names.get('k', ''), self._neumann_markers('k'), dt)
        form = self._add_scalar_transport(
            form, epsilon, psi, wind, d_epsilon,
            self._epsilon_dirichlet_markers(),
            self._neumann_markers('epsilon'), dt)
        # Treat the sink terms as positive implicit contributions.  Lag only
        # their coefficients, following the standard segregated k-epsilon
        # linearization used by production CFD solvers.  Keeping these terms
        # on the right-hand side makes the Picard iteration highly unstable
        # when either turbulence variable changes rapidly.
        form += dt * (epsilon_previous / k_previous) * k * zeta * ngs.dx
        form += dt * self.C_2[time_step] * (
            epsilon_previous / k_previous) * epsilon * psi * ngs.dx
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
            self.C_1[time_step] * epsilon / k * production
            + source_epsilon
        ) * psi * ngs.dx

        if self.DG:
            n = self._normal
            d_k = self.kv[time_step] + nu_t / self.sigma_k[time_step]
            d_epsilon = (
                self.kv[time_step] + nu_t / self.sigma_epsilon[time_step])
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

        for component, test in (('k', zeta), ('epsilon', psi)):
            for marker, values in self.BC.get(
                    'neumann', {}).get(component, {}).items():
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
        scalar_order = max(self.interp_ord - 1, 0)

        for iteration in range(self.nonlinear_max_iters):
            previous.vec.data = gfu.vec
            self.UIter.vec.data = gfu.vec
            self.W[0].vec.data = gfu.components[comp['u']].vec

            if self._wallf is not None:
                self._wallf.update(self.UIter.components[comp['u']])

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

            self._apply_first_cell_epsilon(gfu, time_step)

            if self._limiter is not None:
                self._limiter.bezier_bound(
                    gfu.components[comp['k']],
                    gfu.components[comp['k']].space,
                    scalar_order, (self.k_floor[time_step], 1e20))
                self._limiter.bezier_bound(
                    gfu.components[comp['epsilon']],
                    gfu.components[comp['epsilon']].space,
                    scalar_order, (self.epsilon_floor[time_step], 1e20))

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

        self._apply_first_cell_epsilon(gfu, 0)

        comp = self.model_components
        if self._limiter is not None:
            scalar_order = max(self.interp_ord - 1, 0)
            self._limiter.bezier_bound(
                gfu.components[comp['k']], gfu.components[comp['k']].space,
                scalar_order, (self.k_floor[0], 1e20))
            self._limiter.bezier_bound(
                gfu.components[comp['epsilon']],
                gfu.components[comp['epsilon']].space,
                scalar_order, (self.epsilon_floor[0], 1e20))

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
        if self._limiter is not None:
            comp = self.model_components
            scalar_order = max(self.interp_ord - 1, 0)
            self._limiter.bezier_bound(
                gfu.components[comp['k']], gfu.components[comp['k']].space,
                scalar_order, (self.k_floor[0], 1e20))
            self._limiter.bezier_bound(
                gfu.components[comp['epsilon']],
                gfu.components[comp['epsilon']].space,
                scalar_order, (self.epsilon_floor[0], 1e20))
        super().update_linearization(gfu)
        self.UIter.vec.data = gfu.vec
        if self._wallf is not None:
            self._wallf.update(
                self.UIter.components[self.model_components['u']])
