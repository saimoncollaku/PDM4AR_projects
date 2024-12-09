import random
from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal, RefLaneGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.sim_types import SimTime
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.planning import Trajectory, commands_plan_from_trajectory
from dg_commons.seq.sequence import DgSampledSequence


from pdm4ar.exercises.ex12.visualization import Visualizer
from pdm4ar.exercises.ex12.planner import Planner


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    sg: VehicleGeometry
    sp: VehicleParameters
    planner: Planner

    visualizer: Visualizer
    all_timesteps: list[SimTime]
    all_states: list[VehicleState]

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        assert isinstance(init_obs.goal, RefLaneGoal)
        assert isinstance(init_obs.model_geometry, VehicleGeometry)
        assert isinstance(init_obs.model_params, VehicleParameters)
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params

        self.visualizer = Visualizer(init_obs)
        self.visualizer.set_goal(init_obs.my_name, init_obs.goal, self.sg)

        ptx = init_obs.goal.ref_lane.control_points
        timestamps = list(np.linspace(0, 10, len(ptx)))
        states = [VehicleState(ctr.q.p[0], ctr.q.p[1], ctr.q.theta, 0, 0) for ctr in ptx]
        self.ref_traj = Trajectory(timestamps=timestamps, values=states)

        # initialize planner from planner.py
        # initial state, goal state, dynamics parameters
        # https://github.com/idsc-frazzoli/dg-commons/blob/master/src/dg_commons/sim/models/vehicle.py#L197 <- for dynamics
        self.planner = Planner()

        self.all_timesteps = []
        self.all_states = []

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        my_state = sim_obs.players[self.name].state
        assert isinstance(my_state, VehicleState)

        self.visualizer.plot_scenario(sim_obs)
        self.all_timesteps.append(sim_obs.time)
        self.all_states.append(my_state)

        my_traj = Trajectory(timestamps=self.all_timesteps, values=self.all_states)

        self.visualizer.plot_trajectories([self.ref_traj, my_traj])
        self.visualizer.save_fig()

        # call planner plan
        self.planner.plan()

        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)
