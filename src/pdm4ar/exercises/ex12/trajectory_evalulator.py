# Kinematic check
# Cost calculation
# Collision & trajectory check
# Score candidate trajectories - cost functions, kinematic check & collision check > Behaviour, Velocity, Occlusion Planner

from tkinter.tix import PopupMenu
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.models.vehicle import VehicleState

from pdm4ar.exercises.ex12.sampler.frenet_sampler import Sample
from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference


def scalar_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.stack([x, y], axis=1)
    pure_acceleration = np.linalg.norm(xy, ord=2, axis=1)
    return pure_acceleration


class KinematicsFilter:
    sp: VehicleParameters
    sg: VehicleGeometry
    kappadot_allowed: float
    kappa_allowed: float
    __trajectory: Sample
    reference: np.ndarray

    def __init__(self, sp, sg, reference) -> None:
        self.sp = sp
        self.sg = sg
        self.kappa_allowed = np.tan(self.sp.delta_max) / self.sg.wheelbase
        self.min_vel = 4.0
        self.max_vel = 25.0
        self.reference = reference
        # raise NotImplementedError()

    @property
    def trajectory(self):
        return self.__trajectory

    def check(self, trajectory: Sample):
        self.__trajectory = trajectory
        return {
            "acceleration": self.acceleration_filter(),
            "curvature": self.curvature_filter(),
            "yaw": self.yaw_filter(),
            "delta": self.delta_filter(),
            # "velocity": self.velocity_filter(),
            "goal": self.goal_filter(),
        }

    def acceleration_filter(self):
        """
        a_max check
        there is no v_switch defined in the problem, so not considering

        feasiblity not checked in between the sample times
        """
        pure_acceleration = scalar_value(self.__trajectory.xdotdot, self.__trajectory.ydotdot)
        max_acceleration = np.max(pure_acceleration)
        min_acceleration = np.min(pure_acceleration)
        if self.sp.acc_limits[0] > min_acceleration or self.sp.acc_limits[1] < max_acceleration:
            return False
        return True

    def curvature_filter(self):
        max_kappa = np.max(np.abs(self.__trajectory.kappa))
        # max_kappadot = np.max(np.abs(self.__trajectory.kappadot))
        # if self.kappa_allowed < max_kappa or self.kappadot_allowed < max_kappadot:
        if self.kappa_allowed < max_kappa:
            return False
        return True

    def yaw_filter(self):
        """
        Basically kappa within limits
        """
        return True

    def delta_filter(self):
        self.__trajectory.compute_steering(self.sg.wheelbase)
        ddelta = np.gradient(self.__trajectory.delta)
        if np.max(np.abs(ddelta)) > 2 * self.sp.ddelta_max * self.__trajectory.dt:
            return False
        return True

    def velocity_filter(self):
        pure_velocity = scalar_value(self.__trajectory.xdot, self.__trajectory.ydot)
        max_velocity = np.max(pure_velocity)
        min_velocity = np.min(pure_velocity)
        if min_velocity < self.min_vel or max_velocity > self.max_vel:
            return False
        return True

    def goal_filter(self):
        start_pt = np.stack([self.__trajectory.x[0], self.__trajectory.y[0]])
        end_pt = np.stack([self.__trajectory.x[-1], self.__trajectory.y[-1]])
        start_ref_dist = np.min(np.linalg.norm(self.reference - start_pt, ord=2, axis=1))
        end_ref_dist = np.min(np.linalg.norm(self.reference - end_pt, ord=2, axis=1))

        # print(d_start, d_end)
        if start_ref_dist < 1.0 and not end_ref_dist < 1.0:
            # do not deviate from the reference once you reach it
            return False
        return True


class CollisionFilter:
    __trajectory: Sample

    def __init__(self, init_obs, visualize):
        self.name = init_obs.my_name
        self.sg = init_obs.model_geometry
        self.visualize = visualize
        self.dt = 0.1
        self.dist_parallel = (self.sg.lr + self.sg.lf) / 2.0
        self.dist_perp = self.sg.width

    def check(self, trajectory: Sample, sim_obs, obs_acc):
        self.__trajectory = trajectory
        collides = False
        if self.visualize:
            self.fig, self.axes = plt.subplots()
            # self.axes.scatter(self.__trajectory.x, self.__trajectory.y, c="b")
            self.axes.autoscale()
        for player in sim_obs.players:
            if player != self.name:
                collides = self.collision_filter(
                    player, sim_obs.players[player].state, sim_obs.players[player].occupancy, obs_acc[player]["acc"]
                )
                if collides:
                    break
        if self.visualize:
            self.fig.savefig("/workspaces/student-group-pdm4ar-2024hs-ex12-nopteam/out/12/collision_check.png")
            plt.close(self.fig)
        return collides

    def get_box(self, x, y, psi, inflate=0.0):
        # can only assume all cars have same geometry
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        lr, lf, wh = self.sg.lr + inflate, self.sg.lf + inflate, self.sg.w_half + inflate
        box_corners = [
            [
                x - lr * cos_psi + wh * sin_psi,
                y - lr * sin_psi - wh * cos_psi,
            ],
            [
                x - lr * cos_psi - wh * sin_psi,
                y - lr * sin_psi + wh * cos_psi,
            ],
            [
                x + lf * cos_psi - wh * sin_psi,
                y + lf * sin_psi + wh * cos_psi,
            ],
            [
                x + lf * cos_psi + wh * sin_psi,
                y + lf * sin_psi - wh * cos_psi,
            ],
        ]
        return Polygon(box_corners)

    def collision_filter(self, obs_name, obs_state, obs_box, obs_acc):
        obx, oby = obs_box.exterior.xy
        obs_head = np.array([np.cos(obs_state.psi), np.sin(obs_state.psi)])
        obs_perp = np.array([-np.sin(obs_state.psi), np.cos(obs_state.psi)])
        obs_x, obs_y = obs_state.x, obs_state.y

        for i in range(self.__trajectory.T):
            pt_x, pt_y, pt_psi = self.__trajectory.x[i], self.__trajectory.y[i], self.__trajectory.psi[i]

            obs_vec = np.array([pt_x - obs_x, pt_y - obs_y])
            obs_dot = np.dot(obs_vec, obs_head)
            obs_vel = 0 if obs_dot > 0 else obs_state.vx
            obs_acc = 0 if obs_dot > 0 else obs_acc

            self_box = self.get_box(pt_x, pt_y, pt_psi)

            if self.visualize:
                sbx, sby = self_box.exterior.xy
                self.axes.plot(sbx, sby, color="firebrick", alpha=0.4)
                self.axes.fill(sbx, sby, color="firebrick", alpha=i / self.__trajectory.T * 0.4)

            # for j in range(max(0, i - 1), min(self.__trajectory.T, i + 1)):
            # j = i
            obs_x = obs_x + self.__trajectory.dt * (obs_vel + 0.5 * self.__trajectory.dt * obs_acc) * np.cos(
                obs_state.psi
            )
            obs_y = obs_y + self.__trajectory.dt * (obs_vel + 0.5 * self.__trajectory.dt * obs_acc) * np.sin(
                obs_state.psi
            )

            # obs_box_t = Polygon(zip(obs_x, obs_y))
            dist_parallel = abs(obs_dot)
            dist_perp = abs(np.dot(obs_vec, obs_perp))
            # dist = np.linalg.norm(dist_vec, 2)
            if (dist_perp / self.dist_perp) ** 2 + (dist_parallel / self.dist_parallel) ** 2 <= 1:
                # if self_box.intersects(obs_box_t):
                return True

            if self.visualize:
                self.axes.plot(
                    np.array(obx) + obs_x - obs_state.x,
                    np.array(oby) + obs_y - obs_state.y,
                    color="royalblue" if obs_dot <= 0 else "purple",
                    alpha=0.4,
                )
                self.axes.fill(
                    np.array(obx) + obs_x - obs_state.x,
                    np.array(oby) + obs_y - obs_state.y,
                    color=plt.get_cmap("viridis")(i / self.__trajectory.T),
                    alpha=i / self.__trajectory.T * 0.1,
                )
                # self.axes.arrow(obs_x, obs_y, obs_vec[0], obs_vec[1], width=0.1, alpha=0.1)
                # self.axes.arrow(obs_x, obs_y, obs_perp[0], obs_perp[1], width=0.1, alpha=0.1)
                # self.axes.arrow(obs_x, obs_y, obs_head[0], obs_head[1], width=0.1, alpha=0.1)

        return False


class Cost:
    sp: VehicleParameters
    sg: VehicleGeometry
    __trajectory: Sample
    __observations: SimObservations
    v_ref: float = 20.0
    weights: dict
    cost_functions: list

    def __init__(self, init_obs: InitSimObservations, ref_line: np.ndarray, fn_weights: list) -> None:
        self.name = init_obs.my_name
        assert isinstance(init_obs.model_geometry, VehicleGeometry)
        self.sg = init_obs.model_geometry
        self.__reference = ref_line[::100]
        self.weights = {
            self.penalize_acceleration: fn_weights[0],  # 1e1 order
            # self.penalize_closeness_from_obstacles: 0.02,  # 1e-1 order
            self.penalize_closeness_from_obstacles: fn_weights[1],  # 1e-1 order
            self.penalize_deviation_from_reference: fn_weights[2],  # 1e1 order
            self.penalize_jerk: fn_weights[3],  # 1e2 order
            self.penalize_velocity: fn_weights[4],  # 1e2 order
        }
        self.cost_functions = list(self.weights.keys())
        self.dist_parallel = (self.sg.lr + self.sg.lf) / 2.0 + 0.1
        self.dist_perp = self.sg.width + 0.1

    def get_box(self, x, y, psi, inflate=0.0):
        # can only assume all cars have same geometry
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        lr, lf, wh = self.sg.lr + inflate, self.sg.lf + inflate, self.sg.w_half + inflate
        box_corners = [
            [
                x - lr * cos_psi + wh * sin_psi,
                y - lr * sin_psi - wh * cos_psi,
            ],
            [
                x - lr * cos_psi - wh * sin_psi,
                y - lr * sin_psi + wh * cos_psi,
            ],
            [
                x + lf * cos_psi - wh * sin_psi,
                y + lf * sin_psi + wh * cos_psi,
            ],
            [
                x + lf * cos_psi + wh * sin_psi,
                y + lf * sin_psi - wh * cos_psi,
            ],
        ]
        return Polygon(box_corners)

    def set_weights(self, weights):
        self.weights = {
            self.penalize_acceleration: weights[0],  # 1e1 order
            # self.penalize_closeness_from_obstacles: 0.02,  # 1e-1 order
            self.penalize_closeness_from_obstacles: weights[1],  # 1e-1 order
            self.penalize_deviation_from_reference: weights[2],  # 1e1 order
            self.penalize_jerk: weights[3],  # 1e2 order
            self.penalize_velocity: weights[4],  # 1e2 order
        }

    def get(self, trajectory: Sample, sim_obs: SimObservations, obs_acc):
        self.__observations = sim_obs
        self.__trajectory = trajectory
        self.__obs_acc = obs_acc
        cost = {}
        for cost_fn, weight in self.weights.items():
            cost[cost_fn.__name__] = weight * cost_fn()
        return cost

    @property
    def reference(self):
        return self.__reference

    @property
    def observations(self):
        return self.__observations

    @property
    def trajectory(self):
        return self.__trajectory

    def set_vis(self, fig, axes):
        self.fig, self.axes = fig, axes

    def penalize_acceleration(self):
        pure_acceleration = scalar_value(self.__trajectory.xdotdot, self.__trajectory.ydotdot)
        y = np.square(pure_acceleration)
        return simpson(y, dx=self.__trajectory.dt)

    def penalize_jerk(self):
        pure_jerk = scalar_value(self.__trajectory.xdotdotdot, self.__trajectory.ydotdotdot)
        y = np.square(pure_jerk)
        return simpson(y, dx=self.__trajectory.dt)

    def penalize_velocity(self):
        assert self.v_ref is not None
        pure_velocity = scalar_value(self.__trajectory.xdot, self.__trajectory.ydot)
        # penalize over second half of path
        from_idx = int(self.__trajectory.T / 2)
        cost = self.v_ref**2 - (np.sum(np.square(pure_velocity[from_idx:])) / (self.__trajectory.T - from_idx))
        # cost += np.square(pure_velocity[-1] - self.v_ref)
        return cost

    def penalize_deviation_from_reference(self):
        from_idx = int(self.__trajectory.T / 2)
        pts = np.stack([self.__trajectory.x[from_idx:], self.__trajectory.y[from_idx:]], axis=1)
        dist = [np.min(np.linalg.norm(self.__reference - pt, ord=2, axis=1)) for pt in pts]
        return simpson(np.square(dist), dx=self.__trajectory.dt)

    def penalize_closeness_from_obstacles(self):
        cost = 0
        for player in self.__observations.players:
            if player != self.name:
                obs_box = self.__observations.players[player].occupancy
                obs_state = self.__observations.players[player].state
                obx, oby = obs_box.exterior.xy
                obs_head = np.array([np.cos(obs_state.psi), np.sin(obs_state.psi)])
                obs_perp = np.array([-np.sin(obs_state.psi), np.cos(obs_state.psi)])
                obs_x, obs_y = obs_state.x, obs_state.y

                dist = [1e3]
                for i in range(self.__trajectory.T):
                    pt_x, pt_y, pt_psi = self.__trajectory.x[i], self.__trajectory.y[i], self.__trajectory.psi[i]
                    self_box = self.get_box(pt_x, pt_y, pt_psi)

                    obs_vec = np.array([pt_x - obs_x, pt_y - obs_y])
                    obs_dot = np.dot(obs_vec, obs_head)
                    obs_vel = 0 if obs_dot > 0 else obs_state.vx
                    obs_acc = 0 if obs_dot > 0 else self.__obs_acc[player]["acc"]

                    obs_x = obs_x + self.__trajectory.dt * (obs_vel + 0.5 * self.__trajectory.dt * obs_acc) * np.cos(
                        obs_state.psi
                    )
                    obs_y = obs_y + self.__trajectory.dt * (obs_vel + 0.5 * self.__trajectory.dt * obs_acc) * np.sin(
                        obs_state.psi
                    )

                    dist_parallel = abs(obs_dot)
                    dist_perp = abs(np.dot(obs_vec, obs_perp))

                    # obs_box_t = Polygon(zip(obs_x, obs_y))
                    # dist.append(self_box.distance(obs_box_t))
                    ellip_dist = (dist_perp / self.dist_perp) ** 2 + (dist_parallel / self.dist_parallel) ** 2 - 1
                    dist.append(ellip_dist)

                pen = min(dist)
                cost += 1 / (pen**2)

        return cost


class Evaluator:
    kinematics_filter: KinematicsFilter
    collision_filter: CollisionFilter
    trajectory_cost: Cost
    visualize: bool = False

    def __init__(
        self,
        init_obs: InitSimObservations,
        spline_ref: SplineReference,
        sp: VehicleParameters,
        sg: VehicleGeometry,
        visualize: bool,
    ) -> None:
        ref_line = np.column_stack((spline_ref.x, spline_ref.y))
        self.name = init_obs.my_name
        self.kinematics_filter = KinematicsFilter(sp, sg, ref_line[::100])
        self.collision_filter = CollisionFilter(init_obs, visualize)
        self.obs_kin = {}
        self.dt = 0.1

        # acc, obs, ref, jerk, vel
        self.fn_weights = [0.1, 2.0, 2.5, 0.005, 0.002]

        self.trajectory_cost = Cost(init_obs, ref_line, self.fn_weights)
        self.spline_ref = spline_ref

        self.visualize = visualize
        if self.visualize:
            self.fig, self.axes = plt.subplots()
            self.axes.autoscale()
            self.trajectory_cost.set_vis(self.fig, self.axes)

    def get_best_path(self, all_samples: list[Sample], sim_obs: SimObservations):
        costs = -np.ones(len(all_samples))
        for i, sample in enumerate(all_samples):
            costs_dict = self.get_costs(sample, sim_obs)
            # if "penalize_velocity" in costs_dict:
            #     print(costs_dict)
            costs[i] = sum(costs_dict.values())
        path_sort_idx = np.argsort(costs)
        best_path_index = -1
        for path_idx in path_sort_idx:
            if np.isinf(costs[path_idx]):
                continue
            print(
                "Checking path {}: Total cost: {}, {}".format(
                    path_idx, costs[path_idx], self.get_costs(all_samples[path_idx], sim_obs)
                )
            )
            collides = self.collision_filter.check(all_samples[path_idx], sim_obs, self.obs_kin)
            if not collides:
                best_path_index = path_idx
                break
        # print(best_path_index, costs[best_path_index])
        all_samples[best_path_index].collision_free = True

        if self.visualize:
            self.fig.savefig("/workspaces/student-group-pdm4ar-2024hs-ex12-nopteam/out/12/traj_cost.png")
            plt.close(self.fig)
            self.fig, self.axes = plt.subplots()
            self.trajectory_cost.set_vis(self.fig, self.axes)

        return best_path_index, costs

    def update_obs_acc(self, sim_obs):
        for player in sim_obs.players:
            if player != self.name:
                if player not in self.obs_kin:
                    self.obs_kin[player] = {"acc": 0, "prev_vx": sim_obs.players[player].state.vx}
                else:
                    curr_vx = sim_obs.players[player].state.vx
                    self.obs_kin[player]["acc"] = (curr_vx - self.obs_kin[player]["prev_vx"]) / self.dt
                    self.obs_kin[player]["prev_vx"] = curr_vx

    def get_costs(self, trajectory: Sample, sim_obs: SimObservations) -> dict:
        # cartesian_points = self.spline_ref.get_xy(trajectory)
        # trajectory.x = cartesian_points[:, 0]
        # trajectory.y = cartesian_points[:, 1]

        if not isinstance(trajectory.x, np.ndarray):
            self.spline_ref.to_cartesian(trajectory)
        trajectory.compute_derivatives()

        kinematics_passed_dict = self.kinematics_filter.check(trajectory)
        kinematics_passed = all(kinematics_passed_dict.values())
        trajectory.kinematics_feasible_dict = kinematics_passed_dict
        if not kinematics_passed:
            return {"kinematics_cost": np.inf}
        trajectory.kinematics_feasible = True
        # collides = self.collision_filter.check(trajectory, sim_obs)
        # if collides:
        #     return np.inf
        # trajectory.collision_free = True

        cost = self.trajectory_cost.get(trajectory, sim_obs, self.obs_kin)
        trajectory.cost = cost
        return cost
