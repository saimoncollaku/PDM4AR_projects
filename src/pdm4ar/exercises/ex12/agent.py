from dataclasses import dataclass
from math import isclose
import numpy as np
import logging

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

from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference
from pdm4ar.exercises.ex12.sampler.frenet_sampler import FrenetSampler
from pdm4ar.exercises.ex12.sampler.sim_env_coesion import obtain_complete_ref
from pdm4ar.exercises.ex12.sampler.sim_env_coesion import get_lanelet_distances
from pdm4ar.exercises.ex12.controller import BasicController as Controller
from pdm4ar.exercises.ex12.trajectory_evalulator import Evaluator

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.WARNING, format="%(levelname)s: %(message)s")


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
    sampler: FrenetSampler

    road_distances: tuple[float, float, float]
    spline_ref: SplineReference

    visualizer: Visualizer
    all_timesteps: list[SimTime]
    all_states: list[VehicleState]
    controller: Controller
    evaluator: Evaluator

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

        # * SAIMON PARAMS
        self.last_replan_time = 0
        self.replan_t = 99
        self.best_fp = None

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        assert isinstance(init_obs.goal, RefLaneGoal)
        assert isinstance(init_obs.model_geometry, VehicleGeometry)
        assert isinstance(init_obs.model_params, VehicleParameters)
        assert init_obs.dg_scenario
        assert init_obs.dg_scenario.lanelet_network
        logger.warning("Starting new scenario")

        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params

        # self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.visualizer = Visualizer(init_obs)
        self.visualizer.set_goal(init_obs.my_name, init_obs.goal, self.sg)

        reference_points, target_lanelet_id = obtain_complete_ref(init_obs)
        self.road_distances = get_lanelet_distances(init_obs.dg_scenario.lanelet_network, target_lanelet_id)
        self.spline_ref = SplineReference(reference_points, resolution=int(1e5))

        self.planner = Planner()
        self.controller = Controller(self.sp, self.sg)

        self.evaluator = Evaluator(init_obs, self.spline_ref, self.sp, self.sg)

        self.all_timesteps = []
        self.all_states = []
        self.plans = []

    def create_sampler(self, current_state: VehicleState):
        current_cart = np.column_stack((current_state.x, current_state.y))
        current_frenet = self.spline_ref.to_frenet(current_cart)
        road_l = self.road_distances[0]
        road_r = self.road_distances[1]
        road_generic = self.road_distances[2]
        c_d = current_frenet[0][1]
        s0 = current_frenet[0][0]
        # perf_metric: v_diff = np.maximum(self.max_velocity - 25.0, 5.0 - self.min_velocity)
        self.sampler = FrenetSampler(5, 25, road_l, road_r, road_generic, current_state.vx, c_d, 0, 0, s0)

    def reinitialize_sampler(self, current_state: VehicleState):
        current_cart = np.column_stack((current_state.x, current_state.y))
        current_frenet = self.spline_ref.to_frenet(current_cart)
        c_d = current_frenet[0][1]
        s0 = current_frenet[0][0]
        self.sampler.assign_init_pos(s0, c_d, current_state.vx)

    def trigger_replan(self, sim_obs: SimObservations):
        current_state = sim_obs.players[self.name].state
        current_time = float(sim_obs.time)
        assert isinstance(current_state, VehicleState)
        self.reinitialize_sampler(current_state)

        all_samples = self.sampler.get_paths_merge()
        logger.warning("Sampled {} paths".format(len(all_samples)))

        best_path_index, costs = self.evaluator.get_best_path(all_samples, sim_obs)
        min_cost = costs[best_path_index]
        best_path = all_samples[best_path_index]
        # best_path_index, min_cost = self.check_paths(all_samples)
        logger.warning("Least 3 costs: {}".format(list(np.sort(costs)[0:3].round(2))))
        logger.warning(
            "Path {}: cost {:.3f}, kinematics_feasible: {}, collision_free: {}".format(
                best_path_index, min_cost, best_path.kinematics_feasible, best_path.collision_free
            )
        )

        self.replan_t = best_path.t[-1]
        best_path.compute_steering(self.sg.wheelbase)
        ddelta = np.gradient(best_path.delta)
        print("Best path ddelta: ", np.max(np.abs(ddelta)))

        timestamps = list(best_path.t + current_time)
        states = [
            VehicleState(best_path.x[i], best_path.y[i], best_path.psi[i], best_path.vx[i], best_path.delta[i])
            for i in range(best_path.T)
        ]
        states[0] = current_state
        states[-1].psi = self.initial_psi  # assume heading aligned to lane at the end of trajectory
        states[-2].delta = (states[-1].delta + states[-3].delta) / 2  # hacky fix  for delta bump
        self.agent_traj = Trajectory(timestamps, states)
        self.plans.append(self.agent_traj)
        self.controller.set_reference(self.agent_traj)
        self.last_replan_time = current_time

        return all_samples

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
        current_time = float(sim_obs.time)

        self.all_timesteps.append(sim_obs.time)
        self.all_states.append(my_state)

        my_traj = Trajectory(timestamps=self.all_timesteps, values=self.all_states)

        if np.isclose(current_time, 0):
            self.create_sampler(my_state)
            self.initial_psi = my_state.psi

        if np.isclose(current_time, 0) or np.isclose(float(current_time - self.last_replan_time), self.replan_t):
            all_samples = self.trigger_replan(sim_obs)

            self.visualizer.plot_scenario(sim_obs)
            feas_trajectories = []
            feas_idx = [idx for idx, sample in enumerate(all_samples) if sample.cost != np.inf]
            if len(feas_idx) > 0:
                for s_idx in np.random.choice(feas_idx, 50):
                    path = all_samples[s_idx]
                    path.compute_steering(self.sg.wheelbase)
                    timestamps = list(path.t + current_time)
                    states = [
                        VehicleState(path.x[i], path.y[i], path.psi[i], path.vx[i], path.delta[i])
                        for i in range(path.T)
                    ]
                    feas_trajectories.append(Trajectory(timestamps, states))
                self.visualizer.plot_trajectories(feas_trajectories, colors=["royalblue" for traj in feas_trajectories])
                self.visualizer.save_fig("../../out/12/samples_feas" + str(len(self.plans)) + ".png")
                self.visualizer.clear_viz()

            all_trajectories = []
            self.visualizer.plot_scenario(sim_obs)
            for s_idx in np.random.choice(range(len(all_samples)), 50):
                path = all_samples[s_idx]
                path.compute_steering(self.sg.wheelbase)
                timestamps = list(path.t + current_time)
                states = [
                    VehicleState(path.x[i], path.y[i], path.psi[i], path.vx[i], path.delta[i]) for i in range(path.T)
                ]
                all_trajectories.append(Trajectory(timestamps, states))
            self.visualizer.plot_trajectories(all_trajectories, colors=["grey" for traj in all_trajectories])
            self.visualizer.save_fig("../../out/12/samples" + str(len(self.plans)) + ".png")
            self.visualizer.clear_viz()

        self.visualizer.plot_scenario(sim_obs)
        self.visualizer.plot_trajectories(
            [my_traj, *self.plans], colors=["firebrick", *["green" for plan in self.plans]]
        )
        self.visualizer.save_fig()

        # rnd_acc = random.random() * self.params.param1 * 0
        # rnd_ddelta = (random.random() - 0.5) * self.params.param1 * 0

        # return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)
        cmd_acc, cmd_ddelta = self.controller.get_controls(my_state, sim_obs.time)
        self.controller.plot_controller_perf(len(self.plans))
        self.controller.clear_viz()

        return VehicleCommands(acc=cmd_acc, ddelta=cmd_ddelta)

    def check_paths(self, fplist) -> tuple[int, float]:

        MAX_SPEED = 25  # maximum speed [m/s]
        MIN_SPEED = 5
        MAX_ACCEL = self.sp.acc_limits[1]  # maximum acceleration [m/ss]
        MAX_CURVATURE = np.tan(self.sp.delta_max) / self.sg.length  # maximum curvature [1/m]
        # MAX_CURVATURE = 1

        feasibles = []
        for i in range(len(fplist)):
            _, _, _, _, curv = self.spline_ref.to_cartesian(fplist[i])

            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            # if any([v < MIN_SPEED for v in fplist[i].s_d]):
            # continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in curv]):  # Max curvature check
                continue

            feasibles.append(i)

        bestpath = 0
        mincost = np.inf
        # all_samples = [fplist[i] for i in feasibles]
        for i in feasibles:
            if mincost >= fplist[i].cf:
                mincost = fplist[i].cf
                bestpath = i

        return bestpath, mincost
