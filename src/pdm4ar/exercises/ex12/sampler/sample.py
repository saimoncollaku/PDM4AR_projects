from enum import Enum
import numpy as np


class Samplers(Enum):
    FRENET = 0
    DUBINS = 1


class Sample:
    dt: float
    T: int
    x: np.ndarray
    y: np.ndarray
    kappa: np.ndarray
    vx: np.ndarray
    psi: np.ndarray
    delta: np.ndarray

    xdot: np.ndarray
    xdotdot: np.ndarray
    xdotdotdot: np.ndarray
    ydot: np.ndarray
    ydotdot: np.ndarray
    ydotdotdot: np.ndarray
    kappadot: np.ndarray

    d: np.ndarray
    ddot: np.ndarray
    ddotdot: np.ndarray
    ddotdotdot: np.ndarray
    s: np.ndarray
    sdot: np.ndarray
    sdotdot: np.ndarray
    sdotdotdot: np.ndarray

    t: np.ndarray
    kinematics_feasible: bool = False
    kinematics_feasible_dict: dict[str, bool]
    collision_free: bool = False
    cost: dict

    origin: Samplers

    def __init__(self):
        # Lateral
        self.d = None  # type: ignore
        self.ddot = None  # type: ignore
        self.ddotdot = None  # type: ignore
        self.ddotdotdot = None  # type: ignore

        # Longitudinal
        self.s = None  # type: ignore
        self.sdot = None  # type: ignore
        self.sdotdot = None  # type: ignore
        self.sdotdotdot = None  # type: ignore

        self.x = None  # type: ignore
        self.y = None  # type: ignore
        self.vx = None  # type: ignore
        self.psi = None  # type: ignore

        self.delta = None  # type: ignore
        self.kappa = None  # type: ignore

        self.origin = None  # type: ignore

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

    def store_kappa(self):
        kappa_values = np.zeros(len(self.t))
        cart_dx = np.gradient(self.x)
        cart_dy = np.gradient(self.y)
        cart_d2x = np.gradient(cart_dx)
        cart_d2y = np.gradient(cart_dy)

        for i in range(len(self.t)):
            # Compute curvature (kappa) using Cartesian derivatives
            num = np.abs(cart_dx[i] * cart_d2y[i] - cart_dy[i] * cart_d2x[i])
            den = (cart_dx[i] ** 2 + cart_dy[i] ** 2) ** (3 / 2)
            kappa = num / den if den != 0 else 0.0
            kappa_values[i] = kappa

        self.kappa = kappa_values

    def store_vx_theta(self):
        """
        Requires frenet frame variables
        """
        # Compute curvature and orientation for the Cartesian points

        vx_values = np.zeros(len(self.t))
        theta_values = np.zeros(len(self.t))
        cart_dx = np.gradient(self.x)
        cart_dy = np.gradient(self.y)
        self.store_kappa()
        assert isinstance(self.d, np.ndarray)
        assert isinstance(self.sdot, np.ndarray)
        assert isinstance(self.ddot, np.ndarray)

        for i in range(len(self.t)):
            kappa = self.kappa[i]

            # Compute longitudinal velocity (v_x)
            vx = np.sqrt(((1 - kappa * self.d[i]) ** 2) * self.sdot[i] ** 2 + self.ddot[i] ** 2)
            vx_values[i] = vx

            # Compute orientation (theta)
            theta_c = np.arctan2(cart_dy[i], cart_dx[i])  # Direction of Cartesian trajectory
            delta_theta = np.arctan2(self.ddot[i], self.sdot[i] * (1 - kappa * self.d[i]))  # Aθ
            theta = delta_theta + theta_c
            theta_values[i] = theta

        cart_dx = np.gradient(self.x)
        cart_dy = np.gradient(self.y)

        # Compute orientation (theta) for each point
        self.psi = np.arctan2(cart_dy, cart_dx)
        self.vx = vx_values
        return

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
