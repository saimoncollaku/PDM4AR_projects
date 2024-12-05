from dataclasses import dataclass
from sqlite3.dbapi2 import Timestamp
from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters
from matplotlib import pyplot as plt

from pdm4ar.exercises.ex11.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np
import os


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


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

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

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

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params

        self.bounds = None
        for obs in init_sim_obs.dg_scenario.static_obstacles:
            if obs.shape.geom_type == "LineString":
                self.bounds = obs.shape.bounds

        self.planner = SpaceshipPlanner(
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            bounds=self.bounds,
            tolerances=[init_sim_obs.goal.pos_tol, init_sim_obs.goal.vel_tol, init_sim_obs.goal.dir_tol],
        )
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal_state = init_sim_obs.goal.target

        # self.K = self.planner.params.K
        # self.final_state = SpaceshipState(
        #     self.goal_state.x,
        #     self.goal_state.y,
        #     self.goal_state.psi,
        #     self.goal_state.vx,
        #     self.goal_state.vy,
        #     self.goal_state.dpsi,
        #     0,
        #     self.init_state.m,
        # )

        # self.planner.set_initial_reference(self.init_state, self.final_state)

        dock_points = None
        if isinstance(init_sim_obs.goal, DockingTarget):
            dock_points = init_sim_obs.goal.get_landing_constraint_points_fix()

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state, dock_points)

        self.fig, self.ax = plt.subplots(figsize=(36, 25), dpi=120)
        self.ax.set_xlim([self.bounds[0], self.bounds[2]])
        self.ax.set_ylim([self.bounds[1], self.bounds[3]])
        self.savedir = (
            "../../out/11/" + str(len(self.satellites)) + "_" + str(round(self.planets["Namek"].center[1], 2))
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
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)

        self.ax.scatter(current_state.x, current_state.y, c="b", s=512)
        self.ax.scatter(expected_state.x, expected_state.y, c="r", s=512)
        for name, satellite in self.satellites.items():
            planet_name = name.split("/")[0]
            θ = satellite.omega * float(sim_obs.time) + satellite.tau
            Δθ = np.array([np.cos(θ), np.sin(θ)])
            satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
            satellite_k = plt.Circle(satellite_center, satellite.radius, color="green", alpha=1)
            self.ax.add_patch(satellite_k)
        self.fig.savefig(
            self.savedir + "/mismatch.png",
            bbox_inches="tight",
        )

        #
        # TODO: Implement scheme to replan
        #

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds
