import copy

import numpy as np

from pdm4ar.exercises.ex12.sampler.polynomials import Quartic, Quintic

KJ = 0.1
KT = 0.1
KD = 1e10
KLAT = 1.0
KLON = 1.0


class Sample:
    dt: float
    T: int
    x: np.ndarray
    xdot: np.ndarray
    xdotdot: np.ndarray
    xdotdotdot: np.ndarray
    y: np.ndarray
    ydot: np.ndarray
    ydotdot: np.ndarray
    ydotdotdot: np.ndarray
    kappa: np.ndarray
    kappadot: np.ndarray
    vx: np.ndarray
    psi: np.ndarray
    delta: np.ndarray

    t: np.ndarray
    kinematics_feasible: bool = False
    collision_free: bool = False
    cost: dict

    def __init__(self):
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

        self.x = None
        self.y = None
        self.vx = None
        self.psi = None
        self.delta = None
        self.kappa = None

        self.kinematics_feasible = False
        self.collision_free = False
        self.cost = np.inf

        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

    def get_xy_dot(self, cartesian_points: np.ndarray, time_grad: np.ndarray):
        cart_grad: np.ndarray = np.gradient(cartesian_points, axis=0)
        cartesian_vel = np.zeros((len(time_grad), 2))
        cartesian_vel[:, 0] = cart_grad[:, 0] / time_grad
        cartesian_vel[:, 1] = cart_grad[:, 1] / time_grad
        return cartesian_vel

    def get_xy_dotdot(self, cartesian_vel: np.ndarray, time_grad: np.ndarray):
        return self.get_xy_dot(cartesian_vel, time_grad)

    def get_xy_dotdotdot(self, cartesian_acc: np.ndarray, time_grad: np.ndarray):
        return self.get_xy_dot(cartesian_acc, time_grad)

    def get_kappadot(self, time_grad: np.ndarray):
        kappa_grad: np.ndarray = np.gradient(self.kappa)
        return kappa_grad / time_grad

    def compute_steering(
        self,
        wheelbase,
    ):
        if not isinstance(self.delta, np.ndarray):
            dpsi = np.gradient(self.psi)
            self.delta = np.arctan2(dpsi / self.dt, self.vx / wheelbase)

    def compute_derivatives(self):
        time_grad = np.gradient(np.array(self.t))
        cartesian_points = np.stack([self.x, self.y], axis=1)
        cartesian_vel = self.get_xy_dot(cartesian_points, time_grad)
        cartesian_acc = self.get_xy_dotdot(cartesian_vel, time_grad)
        cartesian_jerk = self.get_xy_dotdotdot(cartesian_acc, time_grad)
        self.xdot = cartesian_vel[:, 0]
        self.xdotdot = cartesian_acc[:, 0]
        self.xdotdotdot = cartesian_jerk[:, 0]
        self.ydot = cartesian_vel[:, 1]
        self.ydotdot = cartesian_acc[:, 1]
        self.ydotdotdot = cartesian_jerk[:, 1]
        self.kappadot = self.get_kappadot(time_grad)


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
        self.v_res = 2

        # ! CAN BE CHANGED
        self.dt = 0.2
        self.max_t = 2.6
        self.min_t = 2.0

    # TODO, its possible that we need to make another path maker for low speed
    def get_paths_merge(self) -> list[Sample]:
        self.last_samples = []

        # Lateral sampling
        for di in np.arange(-self.max_road_r, self.max_road_l + self.road_res, self.road_res):

            # Time sampling
            for ti in np.arange(self.min_t, self.max_t, self.dt):
                fp = Sample()

                lat_qp = Quintic(self.c_d, self.c_d_d, self.c_d_dd, di, 0.0, 0.0, ti)

                fp.dt = self.dt
                fp.t = np.arange(0.0, ti, self.dt)
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

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    tfp.cd = KJ * Jp + KT * ti
                    if tfp.d[-1] > self.road_res / 2 or tfp.d[-1] < -self.road_res / 2:
                        tfp.cd += KD

                    # tfp.cd = KJ * Jp + KT * ti + KD * tfp.d[-1] ** 2
                    tfp.cv = KJ * Js + KT * ti
                    tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                    tfp.T = len(tfp.t)

                    self.last_samples.append(tfp)

        return self.last_samples

    def assign_init_speed(self, index: int, replan_time: float = 3.5) -> None:
        i = int(replan_time / self.dt) - 1

        fp = self.last_samples[index]
        self.c_d_d = fp.d_d[i]
        self.c_d_dd = fp.d_dd[i]

    def assign_init_pos(self, s0, c_d, c_speed) -> None:
        self.c_speed = c_speed
        self.c_d = c_d
        self.s0 = s0
