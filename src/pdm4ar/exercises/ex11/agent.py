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

from pdm4ar.exercises.ex11.planner import SpaceshipPlanner, SolverParameters
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.3
    debug: bool = True


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: SpaceshipState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    tf: float
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
        self.replans = 0

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        print("Player initialised...")
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

        planet_names = "_".join(planet for planet in self.planets.keys())
        satellite_names = "_".join(satellite.split("/")[-1] for satellite in self.satellites.keys())
        planet_satellites = planet_names + "_" + satellite_names

        savefile = f"first_trajectory_{planet_satellites}.pkl"
        if MyAgentParams.debug and os.path.exists(savefile):
            print(f"WARNING: Picking up first trajectory from file {savefile}")
            with open(savefile, "rb") as f:
                self.cmds_plan, self.state_traj, self.tf = pickle.load(f)
        else:
            self.cmds_plan, self.state_traj, self.tf = self.planner.compute_trajectory(self.init_state, self.goal, 0)
            if MyAgentParams.debug:
                print("Saving trajectory to file", savefile)
                with open(savefile, "wb") as f:
                    pickle.dump((self.cmds_plan, self.state_traj, self.tf), f)

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
                        This method is called by the simulator at every simulation time step. (0.1 sec)
                        We suggest to perform two tasks here:
                         - Track the computed trajectory (open or closed loop)
                         - Plan a new trajectory if necessary
                         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)
        In simulation time: 14.6, pred_state: (3.6047077096435505, 9.771710755257988), actual_state: (3.3703899100938406, 9.809520385128122)
        In simulation time: 14.6, pred_state: (3.6047077096435505, 9.771710755257988), actual_state: (3.3703899100938406, 9.809520385128122)

        In simulation time: 14.7, pred_state: (3.7848116057044248, 9.757268463407701), actual_state: (3.520021919520757, 9.81561763531205)
        In simulation time: 14.8, pred_state: (3.9638488268086696, 9.743704770542985), actual_state: (3.6696539289476737, 9.821714885495977)
                        Do **not** modify the signature of this method.
        """
        assert isinstance(self.goal, SpaceshipTarget | DockingTarget)
        current_state = sim_obs.players[self.myname].state
        pred_curr_state = self.state_traj.at_interp(sim_obs.time)
        assert isinstance(current_state, SpaceshipState)
        diff = np.linalg.norm(current_state.as_ndarray() - pred_curr_state.as_ndarray(), ord=1)

        time = float(sim_obs.time)

        state_deviation = {
            "x": current_state.x - pred_curr_state.x,
            "y": current_state.y - pred_curr_state.y,
            "psi": current_state.psi - pred_curr_state.psi,
        }

        if MyAgentParams.debug:
            print(f"In simulation time: {time} diff: {diff.round(4)}, state_deviation: {state_deviation}")

        # dont_plan_last_moment = time < (self.tf - 1.0)
        if diff > MyAgentParams.my_tol:
            # if time > self.tf * 0.8 and self.replans < 0:
            self.replans += 1
            if MyAgentParams.debug:
                print(f"\nReplanning {self.replans}th time\n")
            timestamps = np.linspace(time, self.tf, SolverParameters.K)
            # prev_states = np.array([self.state_traj.at_interp(t).as_ndarray().tolist() for t in timestamps]).T
            prev_cmds = np.array([self.cmds_plan.at_interp(t).as_ndarray().tolist() for t in timestamps]).T
            timestamps = np.expand_dims(np.linspace(0, 1, SolverParameters.K), axis=1)
            prev_states = np.squeeze(
                (1 - timestamps) * current_state.as_ndarray() + timestamps * self.goal_state.as_ndarray()
            ).T
            prev_tf = np.array(self.tf - time)

            assert prev_states.shape == (8, SolverParameters.K)
            assert prev_cmds.shape == (2, SolverParameters.K)
            # print("state init trajectory to follow:")
            # print(prev_states[0:2, :].round(6))
            self.cmds_plan, self.state_traj, self.tf = self.planner.compute_trajectory(
                current_state, self.goal, time, prev_states, prev_cmds, prev_tf
            )

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        return cmds
