import logging
import numpy as np


from dg_commons import PlayerName
from dg_commons.sim.goals import RefLaneGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.planning import Trajectory
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.eval.safety import _get_ttc_of_poly_and_state

from pdm4ar.exercises.ex12.visualization import Visualizer

from pdm4ar.exercises.ex12.params import Pdm4arAgentParams
from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference
from pdm4ar.exercises.ex12.trajectory_evalulator import Evaluator
from pdm4ar.exercises.ex12.controller import BasicController as Controller
from pdm4ar.exercises.ex12.sampler.frenet_sampler import FrenetSampler, Sample
from pdm4ar.exercises.ex12.sampler.sim_env_coesion import obtain_complete_ref, get_lanelet_distances

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.WARNING, format="%(levelname)s %(name)s:\t%(message)s")


class Planner:

    lane_psi: float
    replan_count: int
    last_replan_time: float
    replan_in_t: float
    replan_count: int

    my_name: PlayerName

    controller: Controller
    evaluator: Evaluator
    sampler: FrenetSampler
    cmd_acc: float
    cmd_ddelta: float
    road_distances: tuple[float, float, float]

    def __init__(
        self, init_obs: InitSimObservations, sp: VehicleParameters, sg: VehicleGeometry, params: Pdm4arAgentParams
    ) -> None:
        self.replan_count = 0
        self.last_replan_time = 0.0
        self.replan_in_t = params.start_planning_time
        self.lane_psi = None  # set in the first get_commands

        self.cmd_acc = 0
        self.cmd_ddelta = 0
        self.min_ttc = 0

        self.agent_params = params
        self.sp = sp
        self.sg = sg

        self.visualize = True
        self.all_timesteps = []
        self.all_states = []
        self.plans = []

        self.my_name = init_obs.my_name

        assert isinstance(init_obs.goal, RefLaneGoal)
        assert isinstance(init_obs.model_geometry, VehicleGeometry)
        assert isinstance(init_obs.model_params, VehicleParameters)
        assert init_obs.dg_scenario
        assert init_obs.dg_scenario.lanelet_network
        goal = init_obs.goal
        lanelet_network = init_obs.dg_scenario.lanelet_network

        reference_points, target_lanelet_id = obtain_complete_ref(goal, lanelet_network)
        self.road_distances = get_lanelet_distances(lanelet_network, target_lanelet_id)
        self.spline_ref = SplineReference(reference_points, resolution=int(1e5))
        self.reference = np.column_stack((self.spline_ref.x, self.spline_ref.y))[::100]  # TODO Magic

        self.controller = Controller(self.sp, self.sg, self.visualize)
        self.evaluator = Evaluator(
            init_obs, self.spline_ref, self.sp, self.sg, self.visualize, self.agent_params.eval_weights
        )

        if self.visualize:
            self.visualizer = Visualizer(init_obs)
            self.visualizer.set_goal(self.my_name, init_obs.goal, self.sg)

    def create_sampler(self, current_state: VehicleState):
        """
        This works on the assumption that ego initializes with heading along a lane
        also sets lane_psi
        """
        self.lane_psi = current_state.psi
        current_cart = np.column_stack((current_state.x, current_state.y))
        current_frenet = self.spline_ref.to_frenet(current_cart)
        road_l = self.road_distances[0]
        road_r = self.road_distances[1]
        road_generic = self.road_distances[2] * self.agent_params.lane_change_fraction
        s0 = current_frenet[0][0]
        d0 = current_frenet[0][1]
        if d0 > 0:
            road_l = road_generic * round(abs(d0) / road_generic)
            road_r = 0
        else:
            road_l = 0
            road_r = road_generic * round(abs(d0) / road_generic)

        self.sampler = FrenetSampler(
            min_speed=self.agent_params.min_sample_speed,
            max_speed=self.agent_params.max_sample_speed,
            road_width_l=road_l,
            road_width_r=road_r,
            road_res=road_generic,
            starting_d=d0,
            starting_dd=0,
            starting_ddd=0,
            starting_s=s0,
            starting_sd=current_state.vx,
            starting_sdd=0,
            del_t=self.agent_params.sample_delta_time,
            max_t=self.agent_params.max_sample_time,
            min_t=self.agent_params.min_sample_time,
            v_res=self.agent_params.sdot_sample_space,
        )

    def reinitialize_sampler(self, current_state: VehicleState):
        current_cart = np.column_stack((current_state.x, current_state.y))
        current_frenet = self.spline_ref.to_frenet(current_cart)
        d0 = current_frenet[0][1]
        s0 = current_frenet[0][0]
        sdot = current_state.vx * np.cos(current_state.psi - self.lane_psi)
        ddot = current_state.vx * np.sin(current_state.psi - self.lane_psi)
        sdotdot = self.cmd_acc * np.cos(current_state.psi - self.lane_psi)
        ddotdot = self.cmd_acc * np.sin(current_state.psi - self.lane_psi)
        self.sampler.assign_init_kinematics(s0, d0, sdot, ddot, sdotdot, ddotdot)

    def emergency_stop_trajectory(self, init_state: VehicleState, current_time: float, time_steps: int):
        dt = self.agent_params.dt

        max_deceleration = self.sp.acc_limits[0]
        # req_decel = max(max_deceleration, (self.sampler.min_v - init_state.vx) / dt)
        req_decel = max_deceleration

        states = [init_state]
        for step in range(1, time_steps):
            v = max(init_state.vx + req_decel * dt * step, self.sampler.min_v)
            # v = init_state.vx + req_decel * dt * step
            ux, uy = init_state.vx * np.cos(states[-1].psi), init_state.vx * np.sin(states[-1].psi)

            vx, vy = v * np.cos(states[-1].psi), v * np.sin(states[-1].psi)
            ax, ay = req_decel * np.cos(states[-1].psi), req_decel * np.sin(states[-1].psi)

            x, y = (((vx**2) - (ux**2)) / (2 * ax)) + init_state.x, (((vy**2) - (uy**2)) / (2 * ay)) + init_state.y
            # psi = np.arctan2(y - states[-1].y, x - states[-1].x)
            psi = states[-1].psi + dt * v * np.tan(states[-1].delta) / self.sg.wheelbase

            state = VehicleState(
                x=x,
                y=y,
                psi=psi,
                vx=v,
                delta=np.arctan2((self.lane_psi - states[-1].psi) / dt, v / self.sg.wheelbase),
            )
            states.append(state)
        timesteps = list(current_time + dt * np.arange(time_steps))
        return Trajectory(timesteps, states)

    # scenario update once
    def replan(self, sim_obs: SimObservations) -> list[Sample]:
        self.replan_count += 1
        current_state = sim_obs.players[self.my_name].state
        current_time = float(sim_obs.time)

        ref_progress = self.get_ref_progress(current_state)
        # logger.warning("Progress along reference: {:.2f}".format(ref_progress))

        assert isinstance(current_state, VehicleState)

        if (
            self.min_ttc < self.agent_params.max_min_ttc
            and self.replan_in_t > self.agent_params.dt * self.agent_params.emergency_timesteps
        ):
            logger.error("No feasible paths, Entering emergency trajectory")
            timesteps = self.agent_params.emergency_timesteps
            agent_traj = self.emergency_stop_trajectory(current_state, current_time, timesteps)
            self.replan_in_t = timesteps * self.agent_params.dt
            all_samples = None
        else:
            all_samples = self.sampler.get_paths_merge()
            # logger.warning("Sampled {} paths".format(len(all_samples)))

            best_path_index, costs = self.evaluator.get_best_path(all_samples, sim_obs)
            # min_cost = costs[best_path_index]
            best_path = all_samples[best_path_index]
            # best_path_index, min_cost = self.check_paths(all_samples)

            costs = np.sort(costs)
            # logger.warning("Least 3 costs: {:.3f} {:.3f} {:.3f}" % (costs[0], costs[1], costs[2]))  # type: ignore
            # logger.warning(
            #     "Path {}: cost {:.3f}, kinematics_feasible: {}, collision_free: {}"
            #     % (best_path_index, min_cost, best_path.kinematics_feasible, best_path.collision_free)
            # )  # type: ignore
            # logger.warning(f"kinematics_feasible_dict: {best_path.kinematics_feasible_dict}")  # type: ignore

            # start_pt = np.stack([best_path.x[0], best_path.y[0]])
            # start_ref_dist = np.min(np.linalg.norm(self.reference - start_pt, ord=2, axis=1))
            # end_pt = np.stack([best_path.x[-1], best_path.y[-1]])
            # end_ref_dist = np.min(np.linalg.norm(self.reference - end_pt, ord=2, axis=1))
            # logger.warning("Starting ref dist: {:.3f}, Ending ref dist: {:.3f}".format(start_ref_dist, end_ref_dist))

            if ref_progress <= 0.9 and not (best_path.kinematics_feasible and best_path.collision_free):
                logger.error("No feasible paths, Entering emergency trajectory")
                timesteps = self.agent_params.emergency_timesteps
                agent_traj = self.emergency_stop_trajectory(current_state, current_time, timesteps)
                self.replan_in_t = timesteps * self.agent_params.dt
            else:
                # logger.warning("Replanning at {}".format(current_time))

                best_path.compute_steering(self.sg.wheelbase)
                # ddelta = np.gradient(best_path.delta)
                # logger.warning("Best path ddelta max: {:.3f}".format(np.max(np.abs(ddelta))))

                timestamps = list(best_path.t + current_time)
                states = [
                    VehicleState(best_path.x[i], best_path.y[i], best_path.psi[i], best_path.vx[i], best_path.delta[i])
                    for i in range(best_path.T)
                ]
                states[0] = current_state
                states[-1].psi = self.lane_psi  # assume heading aligned to lane at the end of trajectory
                states[-2].delta = (states[-1].delta + states[-3].delta) / 2  # hacky fix for delta bump
                best_agent_traj = Trajectory(timestamps, states)

                agent_traj = best_agent_traj
                self.replan_in_t = best_path.t[-1] if best_path.towards_goal else self.agent_params.replan_del_t

                print("max steering rate: {:.2f}".format(np.max(np.abs(np.gradient(best_path.delta)))))
                # self.replan_in_t = self.agent_params.replan_del_t

        self.plans.append(agent_traj)
        # self.plans.append(best_agent_traj)
        self.controller.set_reference(agent_traj)
        self.last_replan_time = current_time

        return all_samples

    def get_ttc(self, sim_obs: SimObservations):
        self_box = sim_obs.players[self.my_name].occupancy
        self_state = sim_obs.players[self.my_name].state
        min_ttc = 100
        collide_obs = None
        for player, obs in sim_obs.players.items():
            if player != self.my_name:
                ttc, dist1, dist_2 = _get_ttc_of_poly_and_state(self_box, obs.occupancy, self_state, obs.state)
                if ttc < min_ttc:
                    min_ttc = ttc
                    collide_obs = player
        return min_ttc, collide_obs

    def get_ref_progress(self, curr_state):
        curr_pos = np.array([curr_state.x, curr_state.y])
        ref_idx = np.argmin(np.linalg.norm(self.reference - curr_pos, ord=2, axis=1))
        return ref_idx / self.reference.shape[0]

    def get_commands(self, sim_obs: SimObservations):

        current_state = sim_obs.players[self.my_name].state
        assert isinstance(current_state, VehicleState)
        current_time = float(sim_obs.time)

        self.all_timesteps.append(sim_obs.time)
        self.all_states.append(current_state)
        my_traj = Trajectory(timestamps=self.all_timesteps, values=self.all_states)

        self.evaluator.update_obs_acc(sim_obs)

        self.min_ttc, collide_obs = self.get_ttc(sim_obs)
        if collide_obs is not None:
            logger.error("Collision with {} in {:.2f} s".format(collide_obs, self.min_ttc))

        # if self.min_ttc < self.replan_in_t:
        # logger.warning("Collision detected, replanning")

        # self.visualizer.clear_viz()
        # self.visualizer.plot_scenario(sim_obs)
        # self.visualizer.plot_reference(self.reference_points)
        # self.visualizer.clear_viz()

        if self.lane_psi is None:  # runs once
            self.create_sampler(current_state)

        if current_time < self.agent_params.start_planning_time:  # runs till we get some context
            return 0.0, 0.0

        if np.isclose(float(current_time - self.last_replan_time), self.replan_in_t) or (
            self.min_ttc < self.agent_params.max_min_ttc
            and self.replan_in_t > self.agent_params.dt * self.agent_params.emergency_timesteps
        ):
            # or (
            # self.min_ttc < self.agent_params.max_min_ttc
            # and self.replan_in_t > self.agent_params.dt * self.agent_params.emergency_timesteps)
            logger.warning("Replanning at {}".format(current_time))
            self.reinitialize_sampler(current_state)
            all_samples = self.replan(sim_obs)

            if self.visualize:
                self.visualizer.plot_scenario(sim_obs)
                self.visualizer.plot_samples(all_samples, self.sg.wheelbase, len(self.plans))
                self.visualizer.clear_viz()

        if self.visualize:
            self.visualizer.plot_scenario(sim_obs)
            self.visualizer.plot_trajectories(
                [my_traj, *self.plans], colors=["firebrick", *["green" for plan in self.plans]]
            )
            self.visualizer.save_fig()

        self.cmd_acc, self.cmd_ddelta = self.controller.get_controls(current_state, sim_obs.time)

        if self.visualize:
            self.controller.plot_controller_perf(len(self.plans))
            self.controller.clear_viz()
        return self.cmd_acc, self.cmd_ddelta


class Observer:

    def __init__(self, agent_params: Pdm4arAgentParams) -> None:
        self.agent_params = agent_params

    def update_obs_acc(self, sim_obs, my_name):
        observation_dict = {}
        for player in sim_obs.players:
            if player != my_name:
                if player not in observation_dict:
                    observation_dict[player] = {"acc": 0, "prev_vx": sim_obs.players[player].state.vx}
                else:
                    curr_vx = sim_obs.players[player].state.vx
                    observation_dict[player]["acc"] = (
                        curr_vx - observation_dict[player]["prev_vx"]
                    ) / self.agent_params.dt
                    observation_dict[player]["prev_vx"] = curr_vx
