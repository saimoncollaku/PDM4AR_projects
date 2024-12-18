# Kinematic check
# Cost calculation
# Collision & trajectory check
# Score candidate trajectories - cost functions, kinematic check & collision check > Behaviour, Velocity, Occlusion Planner

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

    def __init__(self, sp, sg) -> None:
        self.sp = sp
        self.sg = sg
        self.kappa_allowed = np.tan(self.sp.delta_max) / self.sg.wheelbase
        # raise NotImplementedError()

    @property
    def trajectory(self):
        return self.__trajectory

    def check(self, trajectory: Sample):
        self.__trajectory = trajectory
        return self.acceleration_filter() and self.curvature_filter() and self.yaw_filter()

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


class CollisionFilter:
    __trajectory: Sample

    def __init__(self, init_obs):
        self.name = init_obs.my_name
        self.sg = init_obs.model_geometry

    def check(self, trajectory: Sample, sim_obs):
        self.__trajectory = trajectory
        collides = False
        self.fig, self.axes = plt.subplots()
        # self.axes.scatter(self.__trajectory.x, self.__trajectory.y, c="b")
        self.axes.autoscale()
        for player in sim_obs.players:
            if player != self.name:
                collides = self.collision_filter(sim_obs.players[player].state)
                if collides:
                    break
        self.fig.savefig("../../out/12/collision_check.png")
        plt.close(self.fig)
        return collides

    def get_box(self, x, y, psi):
        # can only assume all cars have same geometry
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        box_corners = [
            [
                x - self.sg.lr * cos_psi + self.sg.w_half * sin_psi,
                y - self.sg.lr * sin_psi - self.sg.w_half * cos_psi,
            ],
            [
                x - self.sg.lr * cos_psi - self.sg.w_half * sin_psi,
                y - self.sg.lr * sin_psi + self.sg.w_half * cos_psi,
            ],
            [
                x + self.sg.lf * cos_psi - self.sg.w_half * sin_psi,
                y + self.sg.lf * sin_psi + self.sg.w_half * cos_psi,
            ],
            [
                x + self.sg.lf * cos_psi + self.sg.w_half * sin_psi,
                y + self.sg.lf * sin_psi - self.sg.w_half * cos_psi,
            ],
        ]
        return Polygon(box_corners)

    def collision_filter(self, obs_state):
        for i in range(self.__trajectory.T):
            pt_x, pt_y = self.__trajectory.x[i], self.__trajectory.y[i]
            self_box = self.get_box(
                pt_x,
                pt_y,
                (
                    np.arctan2(self.__trajectory.y[i + 1] - pt_y, self.__trajectory.x[i + 1] - pt_x)
                    if i < self.__trajectory.T - 1
                    else 0
                ),
            )

            obs_x = obs_state.x + (i * self.__trajectory.dt) * obs_state.vx * np.cos(obs_state.psi)
            obs_y = obs_state.y + (i * self.__trajectory.dt) * obs_state.vx * np.sin(obs_state.psi)
            obs_box = self.get_box(obs_x, obs_y, obs_state.psi)

            # dist = np.linalg.norm(np.array([pt_x - obs_x, pt_y - obs_y]), 2)
            if self_box.intersects(obs_box):
                return True

            sbx, sby = self_box.exterior.xy
            self.axes.plot(sbx, sby, color="firebrick", alpha=0.4)
            self.axes.fill(sbx, sby, color="firebrick", alpha=0.2)

            obx, oby = obs_box.exterior.xy
            self.axes.plot(sbx, sby, color="royalblue", alpha=0.4)
            self.axes.fill(obx, oby, color="royalblue", alpha=0.2)

        return False


class Cost:
    sp: VehicleParameters
    sg: VehicleGeometry
    __trajectory: Sample
    __observations: SimObservations
    v_ref: float = 20.0
    weights: dict
    cost_functions: list

    def __init__(self, init_obs: InitSimObservations, ref_line: np.ndarray) -> None:
        self.name = init_obs.my_name
        ref_line_vec = (ref_line[-1] - ref_line[0]) / np.linalg.norm(ref_line[-1] - ref_line[0], 2)
        self.__reference = (ref_line[0], ref_line_vec)
        self.weights = {
            self.penalize_acceleration: 1.0,  # 1e1 order
            self.penalize_closeness_from_obstacles: 1.0,  # 1e-1 order
            self.penalize_deviation_from_reference: 1.0,  # 1e1 order
            self.penalize_jerk: 1.0,  # 1e2 order
            self.penalize_velocity: 0.0,  # 1e3 order
        }
        self.cost_functions = list(self.weights.keys())

    def get(self, trajectory: Sample, sim_obs: SimObservations):
        self.__observations = sim_obs
        self.__trajectory = trajectory
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
        cost = (self.__trajectory.T - from_idx) * self.v_ref**2 - np.sum(np.square(pure_velocity[from_idx:]))
        # cost += np.square(pure_velocity[-1] - self.v_ref)
        return cost

    def penalize_deviation_from_reference(self):
        ref_pt, ref_vec = self.__reference
        pts = np.stack([self.__trajectory.x, self.__trajectory.y], axis=1)
        dist = np.cross(ref_vec, pts - ref_pt)
        return simpson(np.square(dist), dx=self.__trajectory.dt)

    def penalize_closeness_from_obstacles(self):
        ts = np.linspace(0, self.__trajectory.dt * self.__trajectory.T, self.__trajectory.T)
        player_pos = np.stack([self.__trajectory.x, self.__trajectory.y], axis=1)
        cost = 0

        for player in self.__observations.players:
            if player != self.name:
                obs_state = self.__observations.players[player].state
                assert isinstance(obs_state, VehicleState)
                T = len(ts)
                obs_pos = np.zeros((T, 2))
                obs_pos[:, 0] = obs_state.x + obs_state.vx * np.cos(obs_state.psi) * ts
                obs_pos[:, 1] = obs_state.y + obs_state.vx * np.sin(obs_state.psi) * ts

                dist = np.linalg.norm(obs_pos - player_pos, ord=2, axis=1)
                cost += simpson(1 / (dist**2), dx=self.__trajectory.dt)
        return cost


class Evaluator:
    kinematics_filter: KinematicsFilter
    collision_filter: CollisionFilter
    trajectory_cost: Cost

    def __init__(
        self, init_obs: InitSimObservations, spline_ref: SplineReference, sp: VehicleParameters, sg: VehicleGeometry
    ) -> None:
        self.kinematics_filter = KinematicsFilter(sp, sg)
        self.collision_filter = CollisionFilter(init_obs)
        ref_line = np.column_stack((spline_ref.x, spline_ref.y))
        self.trajectory_cost = Cost(init_obs, ref_line)
        self.spline_ref = spline_ref

    def get_best_path(self, all_samples: list[Sample], sim_obs: SimObservations):
        costs = -np.ones(len(all_samples))
        for i, sample in enumerate(all_samples):
            costs[i] = sum(self.get_costs(sample, sim_obs).values())
        path_sort_idx = np.argsort(costs)
        for path_idx in path_sort_idx:
            print(self.get_costs(all_samples[path_idx], sim_obs))
            collides = self.collision_filter.check(all_samples[path_idx], sim_obs)
            if not collides:
                best_path_index = path_idx
                break
        print(best_path_index, costs[best_path_index])
        all_samples[best_path_index].collision_free = True
        return best_path_index, costs

    def get_costs(self, trajectory: Sample, sim_obs: SimObservations) -> dict:
        # cartesian_points = self.spline_ref.get_xy(trajectory)
        # trajectory.x = cartesian_points[:, 0]
        # trajectory.y = cartesian_points[:, 1]
        time_array = np.array(trajectory.t)
        time_grad = np.gradient(time_array)
        if not isinstance(trajectory.x, np.ndarray):
            cartesian_points, _, _, _, kappa = self.spline_ref.to_cartesian(trajectory)
            trajectory.x = cartesian_points[:, 0]
            trajectory.y = cartesian_points[:, 1]
            trajectory.kappa = kappa
        else:
            cartesian_points = np.stack([trajectory.x, trajectory.y], axis=1)
        cartesian_vel = self.spline_ref.get_xy_dot(cartesian_points, time_grad)
        cartesian_acc = self.spline_ref.get_xy_dotdot(cartesian_vel, time_grad)
        cartesian_jerk = self.spline_ref.get_xy_dotdotdot(cartesian_acc, time_grad)
        trajectory.xdot = cartesian_vel[:, 0]
        trajectory.xdotdot = cartesian_acc[:, 0]
        trajectory.xdotdotdot = cartesian_jerk[:, 0]
        trajectory.ydot = cartesian_vel[:, 1]
        trajectory.ydotdot = cartesian_acc[:, 1]
        trajectory.ydotdotdot = cartesian_jerk[:, 1]
        trajectory.kappadot = self.spline_ref.get_kappadot(trajectory.kappa, time_grad)

        kinematics_passed = self.kinematics_filter.check(trajectory)
        if not kinematics_passed:
            return {"kinematics_cost": np.inf}
        trajectory.kinematics_feasible = True
        # collides = self.collision_filter.check(trajectory, sim_obs)
        # if collides:
        #     return np.inf
        # trajectory.collision_free = True

        cost = self.trajectory_cost.get(trajectory, sim_obs)
        trajectory.cost = cost
        return cost
