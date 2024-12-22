import copy

import numpy as np

from pdm4ar.exercises.ex12.sampler.polynomials import Quartic, Quintic


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
    kinematics_feasible_dict: dict[str, bool]
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

        self.x = None  # type: ignore
        self.y = None  # type: ignore
        self.vx = None  # type: ignore
        self.psi = None  # type: ignore
        self.delta = None  # type: ignore
        self.kappa = None  # type: ignore

        self.kinematics_feasible = False
        self.collision_free = False
        self.cost = {}

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
        starting_d: float,
        starting_dd: float,
        starting_ddd: float,
        starting_s: float,
        starting_sd: float,
        starting_sdd: float,
        dt: float = 0.1,
        max_t: float = 2.6,
        min_t: float = 2.0,
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
        self.max_t = max_t
        self.min_t = min_t

    def get_paths_merge(self) -> list[Sample]:
        self.last_samples = []

        # Lateral sampling
        for di in np.arange(-self.max_road_r, self.max_road_l + self.road_res, self.road_res):

            # Time sampling
            for ti in np.arange(self.min_t, self.max_t, self.dt):
                fp = Sample()

                lat_qp = Quintic(self.d0, self.ddot, self.ddotdot, di, 0.0, 0.0, ti)

                fp.dt = self.dt
                fp.t = np.arange(0.0, ti, self.dt)
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Loongitudinal sampling
                for vi in np.arange(self.min_v, self.max_v, self.v_res):
                    tfp = copy.deepcopy(fp)
                    lon_qp = Quartic(self.s0, self.sdot, self.sdotdot, vi, 0.0, ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    tfp.T = len(tfp.t)

                    self.last_samples.append(tfp)

        return self.last_samples

    def assign_init_kinematics(self, s0, d0, sdot, ddot, sdotdot, ddotdot) -> None:
        self.sdot = sdot
        self.d0 = d0
        self.s0 = s0
        self.ddot = ddot
        self.sdotdot = sdotdot
        self.ddotdot = ddotdot
