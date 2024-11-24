from dataclasses import dataclass, field
from re import M
import time
from typing import Union, Sequence

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        bounds: Sequence[StaticObstacle],
        tolerances: Sequence[float],
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        self.bounds = bounds
        # print(bounds)
        self.pos_tol, self.vel_tol, self.dir_tol = tolerances
        self.r_s = min(
            np.sqrt(self.sg.w_half**2 + self.sg.l_r**2),
            np.sqrt(self.sg.w_half**2 + self.sg.l_f**2),
            self.sg.l_f + self.sg.l_c,
        )

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = SpaceshipState(
            goal_state.x,
            goal_state.y,
            goal_state.psi,
            goal_state.vx,
            goal_state.vy,
            goal_state.dpsi,
            0,
            init_state.m,
        )

        self.problem_parameters["eta"].value = self.params.tr_radius
        self.first_iteration = True

        # set reference trajectory X_bar, U_bar, p_bar
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        for i in range(self.params.max_iterations):
            self._convexification()
            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            print(str(i) + ":  ", self.problem.status)

            if self._check_convergence(
                self.variables["X"].value,
                self.variables["U"].value,
                self.variables["p"].value,
                self.variables["nu_dyn"].value,
                {name: self.variables["nu_" + name] for name in self.planets},
            ):
                print("Converged in {} iterations..".format(i))
                break

            self._update_trust_region()

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array(
            self.variables["p"].value, self.variables["U"].value, self.variables["X"].value
        )

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        # X = np.zeros((self.spaceship.n_x, K))
        # U = np.zeros((self.spaceship.n_u, K))
        # p = np.ones((self.spaceship.n_p))

        X = np.array(
            [
                (1 - i / self.params.K) * self.init_state.as_ndarray()
                + (i / self.params.K) * self.goal_state.as_ndarray()
                for i in range(self.params.K)
            ]
        ).reshape((self.spaceship.n_x, K))
        U = np.array([(0, 0) for i in range(self.params.K)]).reshape((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))  # reasonable guess: assume direct beeline at max thrust

        print(X.shape, U.shape, p.shape)
        # assert X.shape == (self.spaceship.n_x, K)
        # assert U.shape == (self.spaceship.n_u, K)
        # assert p.shape == (self.spaceship.n_p)

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
            "nu_dyn": cvx.Variable((self.spaceship.n_x, self.params.K - 1)),
        }

        for name in self.planets:
            variables["nu_" + str(name)] = cvx.Variable(self.params.K)

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal_state": cvx.Parameter(self.spaceship.n_x),
            "state_bounds": cvx.Parameter(7),
            "tols": cvx.Parameter(3),
            "control_bounds": cvx.Parameter(4),
            "A": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_x, self.params.K - 1)),
            "B_plus": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_u, self.params.K - 1)),
            "B_minus": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_u, self.params.K - 1)),
            "F": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_p, self.params.K - 1)),
            "r": cvx.Parameter((self.spaceship.n_x, self.params.K - 1)),
            "X_ref": cvx.Parameter((self.spaceship.n_x, self.params.K)),
            "U_ref": cvx.Parameter((self.spaceship.n_u, self.params.K)),
            "p_ref": cvx.Parameter((self.spaceship.n_p)),
            "eta": cvx.Parameter(),
            # Jacobians (Eq44)
            # virutal control variables,
            # trust region variables,
            # g_ic, g_tc?
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        constraints = [self.variables["X"][:, 0] == self.problem_parameters["init_state"], self.variables["p"] >= 0]

        boundary_constraints = [
            self.variables["X"][0, :] >= self.problem_parameters["state_bounds"][0],
            self.variables["X"][1, :] >= self.problem_parameters["state_bounds"][1],
            self.variables["X"][0, :] <= self.problem_parameters["state_bounds"][2],
            self.variables["X"][1, :] <= self.problem_parameters["state_bounds"][3],
            self.variables["X"][6, :] >= self.problem_parameters["state_bounds"][4],
            self.variables["X"][6, :] <= self.problem_parameters["state_bounds"][5],
            self.variables["X"][7, :] >= self.problem_parameters["state_bounds"][6],
        ]
        constraints.extend(boundary_constraints)

        control_constraints = [
            self.variables["U"][:, 0] == 0,
            self.variables["U"][:, -1] == 0,
            self.variables["U"][0, :] >= -self.problem_parameters["control_bounds"][0],
            self.variables["U"][0, :] <= self.problem_parameters["control_bounds"][1],
            self.variables["U"][1, :] >= -self.problem_parameters["control_bounds"][2],
            self.variables["U"][1, :] <= self.problem_parameters["control_bounds"][3],
        ]
        constraints.extend(control_constraints)

        docking_constraints = [
            cvx.norm2(self.variables["X"][0, -1] - self.problem_parameters["goal_state"][0])
            <= self.problem_parameters["tols"][0],
            cvx.norm2(self.variables["X"][1, -1] - self.problem_parameters["goal_state"][1])
            <= self.problem_parameters["tols"][0],
            cvx.norm2(self.variables["X"][2, -1] - self.problem_parameters["goal_state"][2])
            <= self.problem_parameters["tols"][2],
            cvx.norm2(self.variables["X"][3, -1] - self.problem_parameters["goal_state"][3])
            <= self.problem_parameters["tols"][1],
            cvx.norm2(self.variables["X"][4, -1] - self.problem_parameters["goal_state"][4])
            <= self.problem_parameters["tols"][1],
        ]
        constraints.extend(docking_constraints)

        # Boundary constraints on state,
        # see Constraints in writeup,
        # dynamics constraints (Eq55),

        for k in range(self.params.K - 1):
            dyn_constraint = (
                self.variables["X"][:, k + 1]
                == self.variables["X"][:, k]
                + self.problem_parameters["A"][:, k].reshape((self.spaceship.n_x, self.spaceship.n_x))
                @ self.variables["X"][:, k]
                + self.problem_parameters["B_plus"][:, k].reshape((self.spaceship.n_x, self.spaceship.n_u))
                @ self.variables["U"][:, k + 1]
                + self.problem_parameters["B_minus"][:, k].reshape((self.spaceship.n_x, self.spaceship.n_u))
                @ self.variables["U"][:, k]
                + self.problem_parameters["F"][:, k].reshape((self.spaceship.n_x, self.spaceship.n_p))
                @ self.variables["p"]
                + self.problem_parameters["r"][:, k]
                + self.variables["nu_dyn"][:, k]
            )
            constraints.append(dyn_constraint)

        # trust region constraints (Eq55)
        trust_constraint = (
            cvx.norm(self.variables["X"] - self.problem_parameters["X_ref"], 1)
            + cvx.norm(self.variables["U"] - self.problem_parameters["U_ref"], 1)
            + cvx.norm(self.variables["p"] - self.problem_parameters["p_ref"], 1)
            <= self.problem_parameters["eta"]
        )
        constraints.append(trust_constraint)

        # virtual control constraints (Eq55),
        for name, planet in self.planets.items():
            center = np.array(planet.center)
            radius = planet.radius + self.r_s
            for k in range(self.params.K):
                C = -2.0 * (self.variables["X"][0:2, k] - center)
                s = ((radius) ** 2) - (cvx.norm2(self.variables["X"][0:2, k] - center) ** 2)
                r_dash = s - C @ self.problem_parameters["X_ref"][0:2, k]
                obs_constraint = C @ self.variables["X"][0:2, k] + r_dash <= self.variables["nu_" + str(name)][k]
                constraints.append(obs_constraint)

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective = self.params.weight_p @ self.variables["p"]
        objective += self.params.lambda_nu * cvx.norm(self.variables["nu_dyn"], 1)
        for name in self.planets:
            objective += self.params.lambda_nu * cvx.norm(self.variables["nu_" + str(name)], 1)

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        self.problem_parameters["A"].value = A_bar
        self.problem_parameters["B_plus"].value = B_plus_bar
        self.problem_parameters["B_minus"].value = B_minus_bar
        self.problem_parameters["F"].value = F_bar
        self.problem_parameters["r"].value = r_bar
        self.problem_parameters["X_ref"].value = self.X_bar
        self.problem_parameters["U_ref"].value = self.U_bar
        self.problem_parameters["p_ref"].value = self.p_bar

        # equation 44 jacobians
        # FOH discretization analog - need to overload integrator calculate discretization
        # actually, can use the same for CDGr' system - obstacles: needed?

        # need to figure out for H0K0l0 and HfKflf - observe that these are constants, requiring no integration. we are done. linear in our case, so constraint is enough.

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.goal_state.as_ndarray()
        self.problem_parameters["tols"].value = np.array([self.pos_tol, self.vel_tol, self.dir_tol])
        self.problem_parameters["state_bounds"].value = np.array(
            [*self.bounds, self.spaceship.sp.delta_limits[0], self.spaceship.sp.delta_limits[1], self.spaceship.sp.m_v]
        )
        self.problem_parameters["control_bounds"].value = np.array(
            [
                self.spaceship.sp.thrust_limits[0],
                self.spaceship.sp.thrust_limits[1],
                self.spaceship.sp.ddelta_limits[0],
                self.spaceship.sp.ddelta_limits[1],
            ]
        )

    def _check_convergence(self, X, U, p, nu_dyn, nu_obs) -> bool:
        """
        Check convergence of SCvx.
        """
        actual_cost = self.J(X, U, p)
        linearized_cost = self.L(p, nu_dyn, nu_obs)

        return actual_cost - linearized_cost < self.params.stop_crit

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        if self.first_iteration:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
            self.first_iteration = False
            return

        J_opt = self.J(self.variables["X"].value, self.variables["U"].value, self.variables["p"].value)
        J_bar = self.J(self.X_bar, self.U_bar, self.p_bar)
        L_opt = self.L(
            self.variables["p"].value,
            self.variables["nu_dyn"].value,
            {name: self.variables["nu_" + str(name)].value for name in self.planets},
        )
        rho = (J_bar - J_opt) / (J_bar - L_opt)

        if rho >= self.params.rho_0:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

        new_eta = self.problem_parameters["eta"].value
        if rho < self.params.rho_0:
            new_eta = max(self.params.min_tr_radius, new_eta / self.params.alpha)
        elif rho < self.params.rho_1:
            new_eta = max(self.params.min_tr_radius, new_eta / self.params.alpha)
        elif rho >= self.params.rho_2:
            new_eta = min(self.params.max_tr_radius, new_eta * self.params.beta)

        self.problem_parameters["eta"].value = new_eta

    @staticmethod
    def _extract_seq_from_array(
        ts, commands, states
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        # ts = (0, 1, 2, 3, 4)
        # # in case my planner returns 3 numpy arrays
        # F = np.array([0, 1, 2, 3, 4])
        # ddelta = np.array([0, 0, 0, 0, 0])
        # cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        # mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # # in case my state trajectory is in a 2d array
        # npstates = np.random.rand(len(ts), 8)
        # states = [SpaceshipState(*v) for v in npstates]
        # mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        # return mycmds, mystates
        timestamps = np.linspace(0, ts, self.params.K)

        cmds_list = [SpaceshipCommands(x[0], x[1]) for x in commands]
        cmds = DgSampledSequence[SpaceshipCommands](timestamps=timestamps, values=cmds_list)

        states_list = [SpaceshipState(*v) for v in states]
        states = DgSampledSequence[SpaceshipState](timestamps=timestamps, values=states_list)

        return cmds, states

    def J(self, X, U, p):
        flow_map = self.integrator.integrate_nonlinear_full(X, U, p)
        slack_cost = [self.params.lambda_nu * np.linalg.norm(X - flow_map, 1)]
        for name, planet in self.planets.items():
            radius = planet.radius + self.r_s
            center = planet.center
            s = (radius**2) - (np.linalg.norm(X[0:2] - center, axis=1) ** 2)
            slack_cost.append(np.linalg.norm(np.maximum(s, 0), 1))
        cost = 0
        for k in range(self.params.K - 1):
            cost += (slack_cost[k] + slack_cost[k + 1]) / (2 * self.params.K)
        cost += p
        return cost

    def L(self, p, nu_dyn, nu_obs):
        # eqns 52-54
        slack_cost = [self.params.lambda_nu * np.linalg.norm(nu_dyn, 1)]
        for name in self.planets:
            slack_cost.append(self.params.lambda_nu * np.linalg.norm(nu_obs[name], 1))
        cost = 0
        for k in range(self.params.K - 1):
            cost += (slack_cost[k] + slack_cost[k + 1]) / (2 * self.params.K)
        cost += p
        return cost
