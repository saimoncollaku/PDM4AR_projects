from dataclasses import dataclass, field
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
from pdm4ar.exercises.ex11.visualization import Visualizer
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
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
    rho_2: float = 0.7  # trust region 2
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
        # self.r_s = 1.2 * max(
        #     np.sqrt(self.sg.w_half**2 + self.sg.l_r**2),
        #     np.sqrt(self.sg.w_half**2 + self.sg.l_f**2),
        #     self.sg.l_f + self.sg.l_c,
        # )
        self.r_s = self.sg.l

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

        self.verbose = True
        self.iteration = 0
        self.visualizer = Visualizer(self.bounds, self.r_s, planets, satellites, self.params)
        self.visualize = True
        self.vis_per_iters = 10
        self.vis_iter = -1

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

        print(self.init_state, self.goal_state)

        self.problem_parameters["eta"].value = self.params.tr_radius

        # set reference trajectory X_bar, U_bar, p_bar
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        while self.iteration < self.params.max_iterations:
            self._convexification()

            constraints = self._get_constraints()
            objective = self._get_objective()
            self.problem = cvx.Problem(objective, constraints)

            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            print(str(self.iteration) + ":  ", self.problem.status)
            print("[t_f]: {}".format(round(self.variables["p"].value[0], 6)))

            if self.problem.status != "infeasible" and self._check_convergence(
                self.variables["nu_dyn"].value,
                {name: self.variables["nu_" + name].value for name in self.planets},
                {name: self.variables["nu_" + name].value for name in self.satellites},
            ):
                print("Converged in {} iterations..".format(self.iteration))
                print(self.variables["p"].value.round(6))
                print(self.variables["X"].value[0:2].round(6))
                print(self.variables["U"].value.round(6))
                if self.verbose:
                    print("[Slack violations]")
                    print(
                        "Dynamics: ",
                    )
                    for state in range(self.spaceship.n_x):
                        print(self.variables["nu_dyn"].value[state, :].round(6))
                    print("Obstacles:, ")
                    for name in self.planets:
                        print(name, self.variables["nu_" + str(name)].value.round(6))
                    for name in self.satellites:
                        print(name, self.variables["nu_" + str(name)].value.round(6))
                break

            if self.visualize and self.iteration % self.vis_per_iters == 0:
                self.visualizer.vis_iter(
                    self.iteration,
                    self.variables["X"].value,
                    self.variables["p"].value,
                )

            self._update_trust_region()
            self.iteration += 1

        # Example data: sequence from array
        timestamps = list(
            np.linspace(0, self.variables["p"].value, self.params.K).reshape(
                self.params.K,
            )
        )

        mycmds, mystates = self._extract_seq_from_array(
            timestamps, self.variables["U"].value.T, self.variables["X"].value.T
        )

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))

        X[0, :] = np.linspace(self.init_state.x, self.goal_state.x, K)
        X[1, :] = np.linspace(self.init_state.y, self.goal_state.y, K)
        X[2, :] = np.linspace(self.init_state.psi, self.goal_state.psi, K)
        X[7, :] = self.init_state.m
        U[:] = 0
        p[0] = 10.0

        # print(X.shape, U.shape, p.shape)
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
            variables["nu_" + str(name)] = cvx.Variable(self.params.K, nonneg=True)
        for name in self.satellites:
            variables["nu_" + str(name)] = cvx.Variable(self.params.K, nonneg=True)
        for name in self.satellites:
            variables["kappa_" + str(name)] = cvx.Variable(self.params.K, nonneg=True)

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal_state": cvx.Parameter(self.spaceship.n_x),
            "A": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_x, self.params.K - 1)),
            "B_plus": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_u, self.params.K - 1)),
            "B_minus": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_u, self.params.K - 1)),
            "F": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_p, self.params.K - 1)),
            "r": cvx.Parameter((self.spaceship.n_x, self.params.K - 1)),
            "X_ref": cvx.Parameter((self.spaceship.n_x, self.params.K)),
            "U_ref": cvx.Parameter((self.spaceship.n_u, self.params.K)),
            "p_ref": cvx.Parameter((self.spaceship.n_p)),
            "eta": cvx.Parameter(nonneg=True),
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
        constraints = [
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            self.variables["p"] >= 0,
            self.variables["X"][0:5, -1] == self.problem_parameters["goal_state"][0:5],
            # ]
            # boundary_constraints = [
            self.variables["X"][0, :] >= self.bounds[0],
            self.variables["X"][1, :] >= self.bounds[1],
            self.variables["X"][0, :] <= self.bounds[2],
            self.variables["X"][1, :] <= self.bounds[3],
            self.variables["X"][6, :] >= self.spaceship.sp.delta_limits[0],
            self.variables["X"][6, :] <= self.spaceship.sp.delta_limits[1],
            self.variables["X"][7, :] >= self.spaceship.sp.m_v,
            # ]
            # constraints.extend(boundary_constraints)
            # control_constraints = [
            self.variables["U"][:, 0] == 0,
            self.variables["U"][:, -1] == 0,
            self.variables["U"][0, :] >= self.spaceship.sp.thrust_limits[0],
            self.variables["U"][0, :] <= self.spaceship.sp.thrust_limits[1],
            self.variables["U"][1, :] >= self.spaceship.sp.ddelta_limits[0],
            self.variables["U"][1, :] <= self.spaceship.sp.ddelta_limits[1],
            # ]
            # constraints.extend(control_constraints)
            # docking_constraints = [
            cvx.norm2(self.variables["X"][0, -1] - self.problem_parameters["goal_state"][0]) <= self.pos_tol,
            cvx.norm2(self.variables["X"][1, -1] - self.problem_parameters["goal_state"][1]) <= self.pos_tol,
            cvx.norm2(self.variables["X"][2, -1] - self.problem_parameters["goal_state"][2]) <= self.dir_tol,
            cvx.norm2(self.variables["X"][3, -1] - self.problem_parameters["goal_state"][3]) <= self.vel_tol,
            cvx.norm2(self.variables["X"][4, -1] - self.problem_parameters["goal_state"][4]) <= self.vel_tol,
        ]
        # constraints.extend(docking_constraints)

        # Boundary constraints on state,
        # see Constraints in writeup,
        # dynamics constraints (Eq55),

        for k in range(self.params.K - 1):
            dyn_constraint = (
                self.variables["X"][:, k + 1]
                == self.problem_parameters["A"][:, k].reshape((self.spaceship.n_x, self.spaceship.n_x))
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

        for k in range(self.params.K - 1):
            reg_constraint = (
                cvx.abs(self.variables["U"][1, k + 1] - self.variables["U"][1, k])
                <= self.spaceship.sp.ddelta_limits[1] / 4
            )
            constraints.append(reg_constraint)

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
            for k in range(self.params.K):
                H = 1 / (planet.radius + self.r_s)
                Δr = self.problem_parameters["X_ref"][0:2, k] - planet.center
                δr = self.variables["X"][0:2, k] - self.problem_parameters["X_ref"][0:2, k]
                ξ = cvx.norm2(H * Δr)
                ζ = H * H * Δr / (cvx.norm2(H * Δr) + 1e-5)
                obs_constraint = ξ + ζ @ δr >= 1 - self.variables["nu_" + str(name)]
                constraints.append(obs_constraint)

            for k in range(self.params.K - 1):
                H = 1 / (planet.radius + self.r_s)
                pt = 0.5 * self.problem_parameters["X_ref"][0:2, k] + 0.5 * self.problem_parameters["X_ref"][0:2, k + 1]
                Δr = pt - planet.center
                δr = self.variables["X"][0:2, k] - pt
                ξ = cvx.norm2(H * Δr)
                ζ = H * H * Δr / (cvx.norm2(H * Δr) + 1e-5)
                obs_constraint = ξ + ζ @ δr >= 1 - self.variables["nu_" + str(name)]
                constraints.append(obs_constraint)

        for name, satellite in self.satellites.items():
            constraints.append(self.variables["kappa_" + str(name)] <= self.r_s - 2 * self.sg.width)
            for k in range(self.params.K):
                planet_name = name.split("/")[0]
                r = satellite.radius + self.r_s - self.variables["kappa_" + str(name)][k]
                t = k / self.params.K
                θ = satellite.omega * self.problem_parameters["p_ref"].value[0] * t + satellite.tau
                Δθ = np.array([np.cos(θ), np.sin(θ)])
                satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
                Δr = self.problem_parameters["X_ref"][0:2, k] - satellite_center
                δr = self.variables["X"][0:2, k] - self.problem_parameters["X_ref"][0:2, k]
                ξ = cvx.norm2(Δr)
                ζ = Δr / (cvx.norm2(Δr) + 1e-5)
                δθ = np.array([-np.sin(θ), np.cos(θ)])
                δp = self.variables["p"] - self.problem_parameters["p_ref"]
                γ = satellite.orbit_r * satellite.omega * δp * t
                δf = δr - γ * δθ
                obs_constraint = ξ + ζ @ δf >= r - self.variables["nu_" + str(name)][k]
                constraints.append(obs_constraint)

        if self.visualize and self.iteration == self.vis_iter:
            self.visualizer.vis_k(
                self.vis_iter, self.problem_parameters["X_ref"].value, self.problem_parameters["p_ref"].value
            )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective = self.params.weight_p @ self.variables["p"]
        objective += self.params.lambda_nu * cvx.norm(self.variables["nu_dyn"], 1)
        objective += self.params.lambda_nu * sum(cvx.sum(self.variables["nu_" + str(name)]) for name in self.planets)
        objective += self.params.lambda_nu * sum(cvx.sum(self.variables["nu_" + str(name)]) for name in self.satellites)

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

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.goal_state.as_ndarray()

    def _check_convergence(self, nu_dyn, nu_planets, nu_satellites) -> bool:
        """
        Check convergence of SCvx.
        """
        actual_cost = self.J(self.X_bar, self.U_bar, self.p_bar)
        linearized_cost = self.L(nu_dyn, nu_planets, nu_satellites)
        if self.verbose:
            print(
                "[Convergence] J = {}, L = {}, J - L = {}".format(
                    round(actual_cost, 6), round(linearized_cost, 6), round(actual_cost - linearized_cost, 6)
                )
            )

        return actual_cost - linearized_cost < self.params.stop_crit

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        J_opt = self.J(self.variables["X"].value, self.variables["U"].value, self.variables["p"].value)
        J_bar = self.J(self.X_bar, self.U_bar, self.p_bar)
        L_opt = self.L(
            self.variables["nu_dyn"].value,
            {name: self.variables["nu_" + str(name)].value for name in self.planets},
            {name: self.variables["nu_" + str(name)].value for name in self.satellites},
        )
        rho = (J_bar - J_opt) / (J_bar - L_opt)
        if self.verbose:
            print(
                "[TR update] J_bar = {}, J_opt = {}, L_opt = {}".format(
                    round(J_bar, 6), round(J_opt, 6), round(L_opt, 6)
                )
            )

        if rho >= self.params.rho_0:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

        new_eta = self.problem_parameters["eta"].value
        if rho < self.params.rho_0:
            new_eta = max(self.params.min_tr_radius, new_eta / self.params.alpha)
            if self.verbose:
                print("[TR update] [REJECT] rho = {}, eta = {}".format(round(rho, 6), round(new_eta, 6)))
        elif rho < self.params.rho_1:
            new_eta = max(self.params.min_tr_radius, new_eta / self.params.alpha)
            if self.verbose:
                print("[TR update] [SHRINK] rho = {}, eta = {}".format(round(rho, 6), round(new_eta, 6)))
        elif rho >= self.params.rho_2:
            new_eta = min(self.params.max_tr_radius, new_eta * self.params.beta)
            if self.verbose:
                print("[TR update] [ENLARGE] rho = {}, eta = {}".format(round(rho, 6), round(new_eta, 6)))
        elif self.verbose:
            print("[TR update] [STABLE] rho = {}, eta = {}".format(round(rho, 6), round(new_eta, 6)))

        self.problem_parameters["eta"].value = new_eta

    @staticmethod
    def _extract_seq_from_array(
        timestamps, commands, states
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        cmds_list = [SpaceshipCommands(x[0], x[1]) for x in commands]
        states_list = [SpaceshipState(*v) for v in states]

        cmds = DgSampledSequence[SpaceshipCommands](timestamps=timestamps, values=cmds_list)
        states = DgSampledSequence[SpaceshipState](timestamps=timestamps, values=states_list)

        return cmds, states

    def J(self, X, U, p):
        φ = self.integrator.integrate_nonlinear_piecewise(X, U, p)
        δ = np.linalg.norm(X - φ, 1)

        # Nonconvex path constraints, equation 39e
        s_p = 0
        for _, param in self.planets.items():
            dist = np.linalg.norm(X[0:2, :].T - param.center, 2, axis=1)
            s_p += np.sum(np.maximum((param.radius + self.r_s) - dist, 0))

            for k in range(self.params.K - 1):
                midpt = 0.5 * X[0:2, k].T + 0.5 * X[0:2, k + 1].T
                dist = np.linalg.norm(midpt - param.center, 2)
                s_p += np.sum(np.maximum((param.radius + self.r_s) - dist, 0))

        s = 0
        for name, param in self.satellites.items():
            planet_name = name.split("/")[0]
            planet_center = self.planets[planet_name].center
            centers = np.zeros((2, self.params.K))
            for k in range(self.params.K):
                theta = param.omega * p[0] * k / self.params.K + param.tau
                centers[:, k] = planet_center + param.orbit_r * np.array([np.cos(theta), np.sin(theta)])
            dist = np.linalg.norm(X[0:2, :].T - centers.T, 2, axis=1)
            s += np.sum(np.maximum((param.radius + self.r_s - self.variables["kappa_" + str(name)].value) - dist, 0))

        b = 0
        if self.verbose:
            print(
                "[J]: defect = {}, planet_obs = {}, satellite_obs = {}, boundary_obs = {}".format(
                    round(δ, 6), round(s_p, 6), round(s, 6), round(b, 6)
                )
            )

        return δ + s_p + s

    def L(self, nu_dyn, nu_planets, nu_satellites):
        # eqns 52-54
        cost = np.linalg.norm(nu_dyn, 1)
        cost += np.sum([nu for nu in nu_planets.values()])
        cost += np.sum([nu for nu in nu_satellites.values()])

        # print(nu_planets, nu_satellites)

        if self.verbose:
            print(
                "[L]: nu_dyn = {}, nu_planets = {}, nu_satellites = {}".format(
                    round(np.linalg.norm(nu_dyn, 1), 6),
                    round(np.sum([nu for nu in nu_planets.values()]), 6),
                    round(np.sum([nu for nu in nu_satellites.values()]), 6),
                    # round(np.sum(nu_bounds), 6),
                )
            )

        return cost
