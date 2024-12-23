import logging
import numpy as np
from dg_commons import SE2Transform

from pdm4ar.exercises.ex12.sampler.frenet_sampler import Sample
from pdm4ar.exercises.ex12.sampler.dubins_algo import Dubins
from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference
from pdm4ar.exercises.ex12.sampler.sample import Samplers


# logger = logging.getLogger(__name__)
# logging.basicConfig(encoding="utf-8", level=logging.WARNING, format="%(levelname)s %(name)s:\t%(message)s")


class DubinSampler:

    def __init__(
        self,
        min_speed: float,  # depends on penalty
        max_speed: float,  # depends on maximum acceleration in circle
        step_speed: float,
        road_width_l: float,
        road_width_r: float,
        s_max: float,  # how far in s to sample
        sample_ds: float,
        sample_dd: float,
        spline_ref: SplineReference,
        lane_psi: float,
        wheel_base: float,
        max_steering_angle: float,
        max_acceleration: float,
        dt: float = 0.1,
    ):
        self.max_road_l = road_width_l
        self.max_road_r = road_width_r
        self.wheel_base = wheel_base

        self.min_v = min_speed
        self.v_max_along_curve = np.sqrt(max_acceleration * wheel_base / np.tan(max_steering_angle))
        # logger.warning("Maximum velocity to satisfy acceleration on curves: %f", self.v_max_along_curve)
        self.max_v = min(self.v_max_along_curve, max_speed)
        self.step_v = step_speed
        self.dt = dt

        self.sample_ds = sample_ds
        self.sample_dd = sample_dd
        self.s_max = s_max
        self.spline_ref = spline_ref
        self.lane_psi = lane_psi
        self.dubins_generator = Dubins(wheel_base, max_steering_angle)
        self.min_radius = self.dubins_generator.min_radius

    def get_paths(self, s0, d0, psi0, v0) -> list[Sample]:
        samples = []

        all_final_d = [0]
        starting_s = s0 + 3 * self.min_radius
        all_final_s = np.arange(starting_s, starting_s + self.s_max, self.sample_ds)
        max_v = max(self.max_v, v0)
        all_traj_v = np.arange(self.min_v, max_v + self.step_v, self.step_v)
        # final_s can only be beyond a diameter of minimum radius to avoid U-turns.
        num_ss, num_ds, num_vv = len(all_final_s), len(all_final_d), len(all_traj_v)
        num_total = num_ss * num_ds * num_vv
        # logger.warning(
        #     "Generating (%d, %d, %d) s and d values, totaling %d trajectories", num_ss, num_ds, num_vv, num_total
        # )

        # Can vary trajectory velocity
        trajectory_velocity = v0
        # logger.warning("Using trajectory generation velocity %f", trajectory_velocity)
        # if trajectory_velocity > self.max_v or trajectory_velocity < self.min_v:
        # logger.error("Velocity bounds: [%f, %f]", self.min_v, max_v)

        for trajectory_velocity in all_traj_v:
            sample_distance = trajectory_velocity * self.dt
            # logger.warning("Sample minimum distance %f", sample_distance)
            for df in all_final_d:
                for sf in all_final_s:

                    states_in_frenet = [(s0, d0), (sf, df)]
                    states_in_cartesian = self.spline_ref.get_xy(states_in_frenet)
                    init_state = states_in_cartesian[0, :]
                    final_state = states_in_cartesian[1, :]

                    init_config = SE2Transform(init_state.tolist(), psi0)
                    final_config = SE2Transform(final_state.tolist(), self.lane_psi)
                    waypoints = self.dubins_generator.compute_path(init_config, final_config, sample_distance)
                    # self.dubins_generator.plot_trajectory(waypoints, init_config, final_config)

                    sample = Sample()
                    sample.dt = self.dt
                    sample.t = np.arange(0, len(waypoints), 1) * self.dt
                    sample.T = len(sample.t)
                    sample.x = np.array([waypoint.p[0] for waypoint in waypoints])
                    sample.y = np.array([waypoint.p[1] for waypoint in waypoints])
                    sample.psi = np.array([waypoint.theta for waypoint in waypoints])
                    sample.vx = np.array([trajectory_velocity for _ in waypoints])
                    sample.store_kappa()
                    samples.append(sample)
                    sample.origin = Samplers.DUBINS

        return samples
