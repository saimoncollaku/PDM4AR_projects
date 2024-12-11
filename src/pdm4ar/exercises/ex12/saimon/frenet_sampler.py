import copy

import numpy as np

from pdm4ar.exercises.ex12.saimon.polynomials import Quartic, Quintic


class Sample:
    def __init__(self):
        self.t = []

        # Lateral
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # Longitudinal
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []


class FrenetSampler:

    def __init__(
        self,
        min_speed: float,
        max_speed: float,
        road_width_l: float,
        road_width_r: float,
        road_res: float,
        starting_speed: float,
        starting_d: float,
        starting_dd: float,
        starting_ddd: float,
        starting_s: float,
    ):
        self.max_road_l = road_width_l
        self.max_road_r = road_width_r
        self.road_res = road_res
        self.last_samples = []
        self.last_best = []

        self.c_speed = starting_speed
        self.c_d = starting_d
        self.c_d_d = starting_dd
        self.c_d_dd = starting_ddd
        self.s0 = starting_s

        self.min_v = min_speed
        self.max_v = max_speed
        self.v_res = 1

        # ! CAN BE CHANGED
        self.dt = 0.1
        self.max_t = 3.5
        self.min_t = 3

    # TODO, its possible that we need to make another path maker for low speed
    def get_paths_merge(self):
        self.last_samples = []

        # Lateral sampling
        for di in np.arange(-self.max_road_r, self.max_road_l + self.road_res, self.road_res):

            # Time sampling
            for ti in np.arange(self.min_t, self.max_t, self.dt):
                fp = Sample()

                lat_qp = Quintic(self.c_d, self.c_d_d, self.c_d_dd, di, 0.0, 0.0, ti)

                fp.t = [t for t in np.arange(0.0, ti, self.dt)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Loongitudinal sampling
                for vi in np.arange(self.min_v, self.max_v, self.v_res):
                    tfp = copy.deepcopy(fp)
                    lon_qp = Quartic(self.s0, self.c_speed, 0.0, vi, 0.0, ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    self.last_samples.append(tfp)

        return self.last_samples

    def assign_init_conditions(self, index: int) -> None:
        """Assign the initial conditions given the optimal path index

        Args:
            index (int): index of the optimal path from the last batch of samples
        """
        # ! CAN BE CHANGED
        replan_time = 0.1
        i = int(replan_time / self.dt)

        fp = self.last_samples[index]
        self.c_speed = fp.s_d[i]
        self.c_d = fp.d[i]
        self.c_d_d = fp.d_d[i]
        self.c_d_dd = fp.d_dd[i]
        self.s0 = fp.s[i]
