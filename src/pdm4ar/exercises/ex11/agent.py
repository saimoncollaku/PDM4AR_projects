import pickle
from dataclasses import dataclass
from typing import Sequence
import os

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters
from matplotlib import pyplot as plt

from pdm4ar.exercises.ex11.planner import SpaceshipPlanner, SolverParameters
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np
import os


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    end_tol: float = 0.5
    max_tol: float = 0.7
    debug: bool = False
    visualise: bool = False
    cache: bool = False


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState
    goal_spaceship_state: SpaceshipState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    tf: float
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    dock_points: Sequence[np.ndarray] | None

    replans: int
    end_replanned: bool

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets
        self.replans = 0

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        assert init_sim_obs.dg_scenario is not None
        assert isinstance(init_sim_obs.model_geometry, SpaceshipGeometry)
        assert isinstance(init_sim_obs.model_params, SpaceshipParameters)
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.goal = init_sim_obs.goal
        target = self.goal.target
        self.goal_state = SpaceshipState(
            target.x,
            target.y,
            target.psi,
            target.vx,
            target.vy,
            target.dpsi,
            0,
            self.init_state.m,
        )

        self.bounds = None
        for obs in init_sim_obs.dg_scenario.static_obstacles:
            if obs.shape.geom_type == "LineString":
                self.bounds = obs.shape.bounds
        assert self.bounds is not None

        self.planner = SpaceshipPlanner(
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            bounds=self.bounds,
            tolerances=[self.goal.pos_tol, self.goal.vel_tol, self.goal.dir_tol],
        )

        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal_state = init_sim_obs.goal.target
        self.goal_spaceship_state = SpaceshipState(
            self.goal_state.x,
            self.goal_state.y,
            self.goal_state.psi,
            self.goal_state.vx,
            self.goal_state.vy,
            self.goal_state.dpsi,
            0,
            self.init_state.m,
        )

        self.dock_points = None
        if isinstance(init_sim_obs.goal, DockingTarget):
            self.dock_points = init_sim_obs.goal.get_landing_constraint_points_fix()
        planet_names = "_".join(planet for planet in self.planets.keys())
        satellite_names = "_".join(satellite.split("/")[-1] for satellite in self.satellites.keys())
        planet_satellites = planet_names + "_" + satellite_names

        savefile = f"first_trajectory_{planet_satellites}.pkl"
        if MyAgentParams.cache and os.path.exists(savefile):
            print(f"WARNING: Picking up first trajectory from file {savefile}")
            with open(savefile, "rb") as f:
                self.cmds_plan, self.state_traj, self.tf = pickle.load(f)
        else:
            self.cmds_plan, self.state_traj, self.tf = self.planner.compute_trajectory(
                self.init_state, self.goal_state, self.dock_points, 0
            )
            if MyAgentParams.cache:
                print("Saving trajectory to file", savefile)
                with open(savefile, "wb") as f:
                    pickle.dump((self.cmds_plan, self.state_traj, self.tf), f)

        self.replans = 0
        self.end_replanned = False

        if MyAgentParams.visualise:
            out_folder_path = "../.."
            self.fig, self.ax = plt.subplots(figsize=(36, 25), dpi=120)
            self.ax.set_xlim([self.bounds[0], self.bounds[2]])
            self.ax.set_ylim([self.bounds[1], self.bounds[3]])
            self.savedir = (
                f"{out_folder_path}/out/11/"
                + str(len(self.satellites))
                + "_"
                + str(round(self.planets["Namek"].center[1], 2))
            )
            for name, planet in self.planets.items():
                planet = plt.Circle(planet.center, planet.radius, color="green")
                self.ax.add_patch(planet)
            if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
            self.fig.savefig(
                self.savedir + "/mismatch.png",
                bbox_inches="tight",
            )

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
            - Track the computed trajectory (open or closed loop)
            - Plan a new trajectory if necessary
            (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)
        Do **not** modify the signature of this method.
        """
        assert isinstance(self.goal, SpaceshipTarget | DockingTarget)
        current_state = sim_obs.players[self.myname].state
        pred_curr_state = self.state_traj.at_interp(sim_obs.time)
        assert isinstance(current_state, SpaceshipState)
        diff = np.linalg.norm(current_state.as_ndarray()[0:2] - pred_curr_state.as_ndarray()[0:2], ord=2)

        time = float(sim_obs.time)

        if MyAgentParams.debug:
            state_deviation = {
                "x": current_state.x - pred_curr_state.x,
                "y": current_state.y - pred_curr_state.y,
                "psi": current_state.psi - pred_curr_state.psi,
            }
            state_deviation = {k: round(val, 5) for k, val in state_deviation.items()}
            print("Sim time: {:.2f} | diff: {:.4f}".format(time, diff) + f" | state_deviation: {state_deviation}")

        if MyAgentParams.visualise:
            for name, satellite in self.satellites.items():
                planet_name = name.split("/")[0]
                θ = satellite.omega * float(sim_obs.time) + satellite.tau
                Δθ = np.array([np.cos(θ), np.sin(θ)])
                satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
                satellite_k = plt.Circle(satellite_center, satellite.radius, color="green", alpha=1)
                self.ax.add_patch(satellite_k)

            self.ax.scatter(current_state.x, current_state.y, c="b", s=512)
            self.ax.scatter(pred_curr_state.x, pred_curr_state.y, c="r", s=512)
            dist2goal = np.linalg.norm(
                np.array([pred_curr_state.x - self.goal_state.x, pred_curr_state.y - self.goal_state.y]), 2
            )

            if dist2goal < 1.0:
                self.fig.savefig(
                    self.savedir + "/mismatch.png",
                    bbox_inches="tight",
                )

        # if diff > MyAgentParams.end_tol:
        handle_ending = time > self.tf - 1.0 and diff > MyAgentParams.end_tol
        handle_chaos = diff > MyAgentParams.max_tol
        if (handle_ending and not self.end_replanned) or (handle_chaos and self.replans < 10):  # do only 1 replan
            # if time > self.tf * 0.8 and self.replans < 0:
            self.end_replanned = True
            self.replans += 1
            if MyAgentParams.debug:
                print(f"\nReplanning {self.replans}th time\n")
            timestamps = np.linspace(time, self.tf, SolverParameters.K)
            # prev_states = np.array([self.state_traj.at_interp(t).as_ndarray().tolist() for t in timestamps]).T
            prev_cmds = np.array([self.cmds_plan.at_interp(t).as_ndarray().tolist() for t in timestamps]).T
            timestamps = np.expand_dims(np.linspace(0, 1, SolverParameters.K), axis=1)
            prev_states = np.squeeze(
                (1 - timestamps) * current_state.as_ndarray() + timestamps * self.goal_spaceship_state.as_ndarray()
            ).T
            prev_tf = np.array(self.tf - time)

            assert prev_states.shape == (8, SolverParameters.K)
            assert prev_cmds.shape == (2, SolverParameters.K)
            # print("state init trajectory to follow:")
            # print(prev_states[0:2, :].round(6))
            self.cmds_plan, self.state_traj, self.tf = self.planner.compute_trajectory(
                current_state, self.goal_state, self.dock_points, time, prev_states, prev_cmds, prev_tf
            )

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        return cmds
