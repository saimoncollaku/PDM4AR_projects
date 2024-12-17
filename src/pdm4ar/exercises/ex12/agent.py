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
        self.past_t = 0
        self.replan_t = 99

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
        x, y, _, _ = self.spline_ref.obtain_reference_traj(reference_points, resolution=1e5)
        self.ref = np.column_stack((x, y))
        self.ln = init_obs.dg_scenario.lanelet_network

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
            self.sampler = FrenetSampler(
                self.sp.vx_limits[0], self.sp.vx_limits[1], road_l, road_r, road_generic, my_state.vx, c_d, 0, 0, s0
            )

        if np.isclose(float(sim_obs.time), 0) or np.isclose(float(sim_obs.time - self.past_t), self.replan_t):
            fp = self.sampler.get_paths_merge()
            # TODO perform feasibility, cost and collision check
            # (iterate through all)
            best_path_index = self.check_paths(fp)
            self.replan_t = fp[best_path_index].t[-1]
            self.past_t = sim_obs.time
            self.sampler.assign_next_init_conditions(best_path_index, self.replan_t)

            cp = self.spline_ref.to_cartesian(fp[best_path_index])

            timestamps = list(cp[1])
            psi_vals = [
                (
                    np.arctan2(cp[0][i + 1][1] - cp[0][i][1], cp[0][i + 1][0] - cp[0][i][0])
                    if i < cp[0].shape[0] - 1 and i > 0
                    else my_state.psi
                )
                for i in range(cp[0].shape[0])
            ]
            states = [
                VehicleState(
                    cp[0][i][0],
                    cp[0][i][1],
                    psi_vals[i],
                    cp[2][i],
                    (
                        np.arctan2((psi_vals[i + 1] - psi_vals[i]) / 0.1, cp[2][i] / self.sg.wheelbase)
                        if i < cp[0].shape[0] - 1 and i > 0
                        else my_state.delta
                    ),
                )
                for i in range(cp[0].shape[0])
            ]
            self.agent_traj = Trajectory(timestamps, states)
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
        self.controller.plot_controller_perf(self.past_t)

        return VehicleCommands(acc=cmd_acc, ddelta=cmd_ddelta)

    def check_paths(self, fplist) -> int:

        MAX_SPEED = self.sp.vx_limits[1]  # maximum speed [m/s]
        MAX_ACCEL = self.sp.acc_limits[1]  # maximum acceleration [m/ss]
        MAX_CURVATURE = np.tan(self.sp.delta_max) / self.sg.length  # maximum curvature [1/m]
        # MAX_CURVATURE = 1

        feasibles = []
        for i in range(len(fplist)):
            _, _, _, _, curv = self.spline_ref.to_cartesian(fplist[i])

            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in curv]):  # Max curvature check
                continue

            feasibles.append(i)

        bestpath = 0
        mincost = np.inf
        # fp = [fplist[i] for i in feasibles]
        for i in feasibles:
            if mincost >= fplist[i].cf:
                mincost = fplist[i].cf
                bestpath = i

        return bestpath


# def plot_lanelets_and_path(lanelet_network: LaneletNetwork, path: np.ndarray, reference_trajectory: np.ndarray) -> None:
#     """
#     Plot all lanelets in the lanelet network, a given path, and a reference trajectory.

#     :param lanelet_network: The LaneletNetwork containing all lanelets.
#     :param path: A numpy array representing the XY path. Shape should be (n, 2).
#     :param reference_trajectory: A numpy array representing the reference trajectory. Shape should be (n, 2).
#     """
#     plt.figure(figsize=(10, 10))

#     # Plot all lanelets
#     for lanelet in lanelet_network.lanelets:
#         center_vertices = lanelet.center_vertices
#         plt.plot(center_vertices[:, 0], center_vertices[:, 1], label=f"Lanelet {lanelet.lanelet_id}", alpha=0.7)
#         plt.scatter(
#             center_vertices[0, 0],
#             center_vertices[0, 1],
#             color="red",
#             label="Lanelet Start" if lanelet.lanelet_id == lanelet_network.lanelets[0].lanelet_id else None,
#         )
#         plt.scatter(
#             center_vertices[-1, 0],
#             center_vertices[-1, 1],
#             color="blue",
#             label="Lanelet End" if lanelet.lanelet_id == lanelet_network.lanelets[0].lanelet_id else None,
#         )

#     # Plot the path
#     plt.plot(path[:, 0], path[:, 1], color="black", linestyle="--", linewidth=2, label="Path")
#     plt.scatter(path[0, 0], path[0, 1], color="green", label="Path Start Point")
#     plt.scatter(path[-1, 0], path[-1, 1], color="orange", label="Path End Point")

#     # Plot the reference trajectory
#     plt.plot(
#         reference_trajectory[:, 0],
#         reference_trajectory[:, 1],
#         color="purple",
#         linewidth=2,
#         linestyle=":",
#         label="Reference Trajectory",
#     )
#     plt.scatter(
#         reference_trajectory[0, 0], reference_trajectory[0, 1], color="purple", marker="s", label="Ref Start Point"
#     )
#     plt.scatter(
#         reference_trajectory[-1, 0], reference_trajectory[-1, 1], color="purple", marker="x", label="Ref End Point"
#     )

#     # Final Plot Settings
#     plt.xlabel("X-coordinate")
#     plt.ylabel("Y-coordinate")
#     plt.title("Lanelets, Path, and Reference Trajectory")
#     plt.legend()
#     plt.grid(True)
#     plt.axis("equal")
#     plt.show()

#     plt.savefig("ciccino")
