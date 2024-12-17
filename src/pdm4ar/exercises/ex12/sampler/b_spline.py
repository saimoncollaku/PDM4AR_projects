from typing import Tuple
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import splprep, splev

from pdm4ar.exercises.ex12.sampler.frenet_sampler import Sample


class SplineReference:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curv: np.ndarray

    def __init__(
        self,
        points: np.ndarray,
        degree: int = 3,
        resolution: int = 100,
    ):
        points = np.array(points)
        tck, _ = splprep(points.T, k=degree, s=0)
        u = np.linspace(0, 1, int(resolution))

        # Evaluate the spline for x and y
        self.x, self.y = splev(u, tck)

        # Yaw
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        self.yaw = np.arctan2(dy, dx)

        # Curvature
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        num = np.abs(dx * d2y - dy * d2x)
        den = (dx**2 + dy**2) ** (3 / 2)
        self.curv = np.zeros_like(self.x, dtype=float)
        mask = den != 0
        self.curv[mask] = num[mask] / den[mask]

    def to_frenet(self, points: np.ndarray) -> np.ndarray:
        """
        Compute Frenet parameters for multiple points

        Args:
            points (np.ndarray): Points to transform [[x1, y1], [x2, y2], ...]

        Returns:
            np.ndarray: Frenet points [[s1, d1], [s2, d2], ...]
        """
        frenet_points = []

        for point in points:
            distances = np.sqrt((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2)
            closest_idx = np.argmin(distances)
            closest_point = np.array([self.x[closest_idx], self.y[closest_idx]])

            s = np.sum(np.sqrt(np.diff(self.x[: closest_idx + 1]) ** 2 + np.diff(self.y[: closest_idx + 1]) ** 2))
            dx = np.gradient(self.x)
            dy = np.gradient(self.y)
            tangent = np.array([dx[closest_idx], dy[closest_idx]])
            tangent_unit = tangent / np.linalg.norm(tangent)

            normal = np.array([-tangent_unit[1], tangent_unit[0]])
            point_vector = point - closest_point
            d = np.dot(point_vector, normal)

            frenet_points.append([s, d])

        return np.array(frenet_points)

    def get_xy(self, sample: Sample):
        cartesian_points = np.zeros((len(sample.t), 2))
        # Precompute cumulative lengths of the reference trajectory
        cumulative_lengths = np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
        cumulative_lengths = np.concatenate(([0], cumulative_lengths))

        # Compute gradients for the reference trajectory
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        for i, (s, d) in enumerate(zip(sample.s, sample.d)):
            # Find the closest point on the reference trajectory for the given s
            idx = np.argmin(np.abs(cumulative_lengths - s))
            if idx < len(self.x) - 1:
                base_point = np.array([self.x[idx], self.y[idx]])
            else:
                base_point = np.array([self.x[-1], self.y[-1]])

            # Compute tangent and normal vectors
            tangent = np.array([dx[idx], dy[idx]])
            tangent_unit = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent_unit[1], tangent_unit[0]])

            # Compute Cartesian point
            cartesian_point = base_point + d * normal
            cartesian_points[i, :] = cartesian_point
        return cartesian_points

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

    def get_kappadot(self, kappa: np.ndarray, time_grad: np.ndarray):
        kappa_grad: np.ndarray = np.gradient(kappa)
        return kappa_grad / time_grad

    def to_cartesian(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Frenet coordinates (s, d) back to Cartesian coordinates (x, y),
        and compute longitudinal velocity (v_x), orientation (theta), and curvature (kappa).

        Args:
            sample (Sample): A single Sample object containing Frenet trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Cartesian points [[x1, y1], [x2, y2], ...]
            - Corresponding timestamps
            - Longitudinal velocities (v_x) for each time step
            - Orientations (theta) for each time step
            - Curvatures (kappa) for each time step
        """
        vx_values = np.zeros(len(sample.t))
        theta_values = np.zeros(len(sample.t))
        kappa_values = np.zeros(len(sample.t))

        cartesian_points = self.get_xy(sample)

        # Compute curvature and orientation for the Cartesian points
        cart_dx = np.gradient(cartesian_points[:, 0])
        cart_dy = np.gradient(cartesian_points[:, 1])
        cart_d2x = np.gradient(cart_dx)
        cart_d2y = np.gradient(cart_dy)

        for i in range(len(sample.t)):
            # Compute curvature (kappa) using Cartesian derivatives
            num = np.abs(cart_dx[i] * cart_d2y[i] - cart_dy[i] * cart_d2x[i])
            den = (cart_dx[i] ** 2 + cart_dy[i] ** 2) ** (3 / 2)
            kappa = num / den if den != 0 else 0.0
            kappa_values[i] = kappa

            # Compute longitudinal velocity (v_x)
            vx = np.sqrt(((1 - kappa * sample.d[i]) ** 2) * sample.s_d[i] ** 2 + sample.d_d[i] ** 2)
            vx_values[i] = vx

            # Compute orientation (theta)
            theta_c = np.arctan2(cart_dy[i], cart_dx[i])  # Direction of Cartesian trajectory
            delta_theta = np.arctan2(sample.d_d[i], sample.s_d[i] * (1 - kappa * sample.d[i]))  # Aθ
            theta = delta_theta + theta_c
            theta_values[i] = theta

        cart_dx = np.gradient(cartesian_points[:, 0])
        cart_dy = np.gradient(cartesian_points[:, 1])

        # Compute orientation (theta) for each point
        theta_values = np.arctan2(cart_dy, cart_dx)
        sample.x = cartesian_points[:, 0]
        sample.y = cartesian_points[:, 1]
        sample.kappa = kappa_values

        return (
            cartesian_points,
            sample.t,
            vx_values,
            theta_values,
            kappa_values,
        )
