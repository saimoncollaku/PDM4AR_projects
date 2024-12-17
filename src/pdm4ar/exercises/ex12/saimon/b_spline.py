from typing import Tuple
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from pdm4ar.exercises.ex12.saimon.frenet_sampler import FrenetSampler


class SplineReference:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.curv = []

    def obtain_reference_traj(
        self,
        points: np.ndarray,
        degree: int = 3,
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        return self.x, self.y, self.yaw, self.curv

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

    def to_cartesian(self, sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        cartesian_points = []
        vx_values = []
        theta_values = []
        kappa_values = []

        # Precompute cumulative lengths of the reference trajectory
        cumulative_lengths = np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
        cumulative_lengths = np.concatenate(([0], cumulative_lengths))

        # Compute gradients for the reference trajectory
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        for s, d, d_d, s_d in zip(sample.s, sample.d, sample.d_d, sample.s_d):
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
            cartesian_points.append(cartesian_point)

        # Compute curvature and orientation for the Cartesian points
        cartesian_points = np.array(cartesian_points)
        cart_dx = np.gradient(cartesian_points[:, 0])
        cart_dy = np.gradient(cartesian_points[:, 1])
        cart_d2x = np.gradient(cart_dx)
        cart_d2y = np.gradient(cart_dy)

        for i, (x, y, d_d, s_d) in enumerate(
            zip(cartesian_points[:, 0], cartesian_points[:, 1], sample.d_d, sample.s_d)
        ):
            # Compute curvature (kappa) using Cartesian derivatives
            num = np.abs(cart_dx[i] * cart_d2y[i] - cart_dy[i] * cart_d2x[i])
            den = (cart_dx[i] ** 2 + cart_dy[i] ** 2) ** (3 / 2)
            kappa = num / den if den != 0 else 0.0
            kappa_values.append(kappa)

            # Compute longitudinal velocity (v_x)
            vx = np.sqrt(((1 - kappa * d) ** 2) * s_d**2 + d_d**2)
            vx_values.append(vx)

            # Compute orientation (theta)
            theta_c = np.arctan2(cart_dy[i], cart_dx[i])  # Direction of Cartesian trajectory
            delta_theta = np.arctan2(d_d, s_d * (1 - kappa * d))  # Aθ
            theta = delta_theta + theta_c
            theta_values.append(theta)

        cart_dx = np.gradient(cartesian_points[:, 0])
        cart_dy = np.gradient(cartesian_points[:, 1])

        # Compute orientation (theta) for each point
        theta_values = np.arctan2(cart_dy, cart_dx)
        return (
            np.array(cartesian_points),
            np.array(sample.t),
            np.array(vx_values),
            np.array(theta_values),
            np.array(kappa_values),
        )
