import random
from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np
from typing import List, Tuple

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
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
import matplotlib.pyplot as plt

# * SAIMON IMPORTS
from pdm4ar.exercises.ex12.saimon.b_spline import SplineReference
from pdm4ar.exercises.ex12.saimon.frenet_sampler import FrenetSampler
from pdm4ar.exercises.ex12.saimon.sim_env_coesion import obtain_complete_ref
from pdm4ar.exercises.ex12.saimon.sim_env_coesion import get_lanelet_distances
from pdm4ar.exercises.ex12.controller import BasicController as Controller


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

        # * SAIMON PARAMS
        self.road = None
        self.spline_ref = None
        self.samplet = None

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

        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.visualizer = Visualizer(init_obs)
        self.visualizer.set_goal(init_obs.my_name, init_obs.goal, self.sg)

        # * SAIMON TEST #######################################################
        reference_points, target_lanelet_id = obtain_complete_ref(init_obs)
        self.road = get_lanelet_distances(init_obs.dg_scenario.lanelet_network, target_lanelet_id)
        self.spline_ref = SplineReference()
        self.spline_ref.obtain_reference_traj(reference_points, resolution=1e7)

        # initialize planner from planner.py
        # initial state, goal state, dynamics parameters
        # https://github.com/idsc-frazzoli/dg-commons/blob/master/src/dg_commons/sim/models/vehicle.py#L197 <- for dynamics
        self.planner = Planner()
        self.controller = Controller(self.sp, self.sg)

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

        self.all_timesteps.append(sim_obs.time)
        self.all_states.append(my_state)

        my_traj = Trajectory(timestamps=self.all_timesteps, values=self.all_states)

        if np.isclose(float(sim_obs.time), 0):
            current_cart = np.column_stack((my_state.x, my_state.y))
            current_frenet = self.spline_ref.to_frenet(current_cart)
            road_l = self.road["distance_to_leftmost"]
            road_r = self.road["distance_to_rightmost"]
            road_generic = self.road["distance_to_other_lanelet"]
            c_d = current_frenet[0][1]
            s0 = current_frenet[0][0]
            self.sampler = FrenetSampler(10, 50, road_l, road_r, road_generic, my_state.vx, c_d, 0, 0, s0)

        if np.isclose(float(sim_obs.time) % 3.5, 0):
            fp = self.sampler.get_paths_merge()
            # TODO perform feasibility, cost and collision check
            # (iterate through all)
            num_traj = len(self.sampler.last_samples)
            best_path_index = 15
            print(num_traj, best_path_index)

            cp = self.spline_ref.to_cartesian(fp[best_path_index])

            timestamps = list(cp[1])
            states = [
                VehicleState(
                    cp[0][i][0],
                    cp[0][i][1],
                    (
                        np.arctan2(cp[0][i + 1][1] - cp[0][i][1], cp[0][i + 1][0] - cp[0][i][0])
                        if i < cp[0].shape[0] - 1
                        else my_state.psi
                    ),
                    cp[2][i],
                    0,
                )
                for i in range(cp[0].shape[0])
            ]
            self.agent_traj = Trajectory(timestamps, states)

            # TODO find best path
            self.sampler.assign_next_init_conditions(best_path_index)
            self.controller.set_reference(self.agent_traj)

            # self.visualizer.plot_scenario(sim_obs)
            # trajectories = []
            # for sample in np.random.choice(num_traj, 10):
            #     cp = self.spline_ref.to_cartesian(self.sampler.last_samples[sample])
            #     timestamps = list(cp[1])
            #     states = [
            #         VehicleState(
            #             cp[0][i][0],
            #             cp[0][i][1],
            #             (
            #                 np.arctan2(cp[0][i + 1][1] - cp[0][i][1], cp[0][i + 1][0] - cp[0][i][0])
            #                 if i < cp[0].shape[0] - 1
            #                 else my_state.psi
            #             ),
            #             cp[2][i],
            #             0,
            #         )
            #         for i in range(cp[0].shape[0])
            #     ]
            #     trajectories.append(Trajectory(timestamps, states))
            # self.visualizer.plot_trajectories(trajectories, colors=None)
            # self.visualizer.save_fig("../../out/12/samples" + str(round(float(sim_obs.time), 2)) + ".png")
            # self.visualizer.clear_viz()

        self.visualizer.plot_scenario(sim_obs)
        self.visualizer.plot_trajectories([my_traj, self.agent_traj], colors=["firebrick", "green"])
        self.visualizer.save_fig()

        # rnd_acc = random.random() * self.params.param1 * 0
        # rnd_ddelta = (random.random() - 0.5) * self.params.param1 * 0

        # return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)
        cmd_acc, cmd_ddelta = self.controller.get_controls(my_state, sim_obs.time)

        return VehicleCommands(acc=cmd_acc, ddelta=cmd_ddelta)
