import copy
import logging
import numpy as np

from pdm4ar.exercises.ex12.sampler.polynomials import Quartic, Quintic
from pdm4ar.exercises.ex12.sampler.sample import Sample, Samplers


logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.WARNING, format="%(levelname)s %(name)s:\t%(message)s")


class FrenetSampler:

    def __init__(
        self,
        min_speed: float,
        max_speed: float,
        road_width_l: float,
        road_width_r: float,
        road_res: float,
        dt: float = 0.1,
        max_t: float = 2.6,
        min_t: float = 2.0,
        v_res: float = 2,
    ):
        self.max_road_l = road_width_l
        self.max_road_r = road_width_r
        self.road_res = road_res

        self.min_v = min_speed
        self.max_v = max_speed
        self.v_res = v_res

        self.dt = dt
        self.max_t = max_t
        self.min_t = min_t

    def get_paths(self, s0, sdot, sdotdot, d0, ddot, ddotdot) -> list[Sample]:
        last_samples = []

        all_final_d = np.arange(-self.max_road_r, self.max_road_l + self.road_res, self.road_res)
        all_final_t = np.arange(self.min_t, self.max_t, self.dt)
        all_final_v = np.arange(self.min_v, self.max_v, self.v_res)
        num_ts, num_ds, num_vs = len(all_final_t), len(all_final_d), len(all_final_v)

        num_total = num_ts * num_ds * num_vs
        logger.warning(
            "Generating (%d, %d, %d) t, d, v values, totaling %d trajectories", num_ts, num_ds, num_vs, num_total
        )

        # Lateral sampling
        for di in all_final_d:

            # Time sampling
            for ti in all_final_t:
                sample = Sample()
                sample.origin = Samplers.FRENET

                lat_qp = Quintic(d0, ddot, ddotdot, di, 0.0, 0.0, ti)

                sample.dt = self.dt
                sample.t = np.arange(0.0, ti, self.dt)
                sample.d = np.array([lat_qp.calc_point(t) for t in sample.t])
                sample.ddot = np.array([lat_qp.calc_first_derivative(t) for t in sample.t])
                sample.ddotdot = np.array([lat_qp.calc_second_derivative(t) for t in sample.t])
                sample.ddotdotdot = np.array([lat_qp.calc_third_derivative(t) for t in sample.t])

                # Longitudinal sampling
                for vi in all_final_v:
                    copied_sample = copy.deepcopy(sample)
                    lon_qp = Quartic(s0, sdot, sdotdot, vi, 0.0, ti)

                    copied_sample.s = np.array([lon_qp.calc_point(t) for t in sample.t])
                    copied_sample.sdot = np.array([lon_qp.calc_first_derivative(t) for t in sample.t])
                    copied_sample.sdotdot = np.array([lon_qp.calc_second_derivative(t) for t in sample.t])
                    copied_sample.sdotdotdot = np.array([lon_qp.calc_third_derivative(t) for t in sample.t])

                    copied_sample.T = len(copied_sample.t)

                    last_samples.append(copied_sample)

        return last_samples
