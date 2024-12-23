import numpy as np
from dg_commons import SE2Transform

from pdm4ar.exercises.ex12.sampler.frenet_sampler import Sample
from pdm4ar.exercises.ex12.sampler.dubins_algo import Dubins
from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference


class DubinSampler:

    def __init__(
        self,
        min_speed: float,  # depends on penalty
        max_speed: float,  # depends on maximum acceleration in circle
        road_width_l: float,
        road_width_r: float,
        max_s: float,  # how far in s to sample
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

        self.min_v = min_speed
        self.max_v = max_speed
        self.dt = dt

        self.sample_ds = sample_ds
        self.sample_dd = sample_dd
        self.max_s = max_s
        self.spline_ref = spline_ref
        self.lane_psi = lane_psi
        self.dubins_generator = Dubins(wheel_base, max_steering_angle)
        self.min_radius = self.dubins_generator.min_radius
        self.v_max_along_curve = np.sqrt(max_acceleration * wheel_base / np.tan(max_steering_angle))

    def get_paths(self, s0, d0, psi0) -> list[Sample]:
        samples = []

        for df in np.arange(-self.max_road_r, self.max_road_l + self.sample_dd):
            for sf in np.arange(s0 + 2 * self.min_radius, s0 + 2 * self.min_radius + self.max_s, self.sample_ds):
                states_in_frenet = [(s0, d0), (sf, df)]
                states_in_cartesian = self.spline_ref.get_xy(states_in_frenet)
                init_state = states_in_cartesian[0, :]
                final_state = states_in_cartesian[1, :]
                init_config = SE2Transform(init_state.tolist(), psi0)
                final_config = SE2Transform(final_state.tolist(), self.lane_psi)
                self.dubins_generator.compute_path(init_config, final_config)
        return samples
