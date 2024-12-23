import numpy as np
from dg_commons import SE2Transform

from pdm4ar.exercises.ex12.sampler.frenet_sampler import Sample
from pdm4ar.exercises.ex12.sampler.dubins_algo import calculate_dubins_path


class DubinSampler:

    def __init__(
        self,
        min_speed: float,
        max_speed: float,
        road_width_l: float,
        road_width_r: float,
        road_res: float,
        starting_d: float,
        starting_dd: float,
        starting_ddd: float,
        starting_s: float,
        starting_sd: float,
        starting_sdd: float,
        dt: float = 0.1,
        v_res: float = 2,
    ):
        self.max_road_l = road_width_l
        self.max_road_r = road_width_r
        self.road_res = road_res
        self.last_samples = []
        self.last_best = []

        self.d0 = starting_d
        self.ddot = starting_dd
        self.ddotdot = starting_ddd
        self.s0 = starting_s
        self.sdot = starting_sd
        self.sdotdot = starting_sdd

        self.min_v = min_speed
        self.max_v = max_speed
        self.v_res = v_res

        self.dt = dt
