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

        tck, _ = splprep(points.T, k=degree, s=1)
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

    # def to_cartesian(self, frenet_points: np.ndarray) -> np.ndarray:
    #     """
    #     Convert Frenet coordinates (s, d) back to Cartesian coordinates (x, y)

    #     Args:
    #         frenet_points (np.ndarray): Frenet points [[s1, d1], [s2, d2], ...]

    #     Returns:
    #         np.ndarray: Cartesian points [[x1, y1], [x2, y2], ...]
    #     """
    #     cartesian_points = []

    #     for s, d in frenet_points:
    #         cumulative_lengths = np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
    #         idx = np.argmin(np.abs(cumulative_lengths - s))
    #         if idx < len(self.x) - 1:
    #             base_point = np.array([self.x[idx], self.y[idx]])
    #         else:
    #             base_point = np.array([self.x[-1], self.y[-1]])

    #         dx = np.gradient(self.x)
    #         dy = np.gradient(self.y)
    #         tangent = np.array([dx[idx], dy[idx]])
    #         tangent_unit = tangent / np.linalg.norm(tangent)
    #         normal = np.array([-tangent_unit[1], tangent_unit[0]])

    #         cartesian_point = base_point + d * normal
    #         cartesian_points.append(cartesian_point)

    #     return np.array(cartesian_points)
    def to_cartesian(self, sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Frenet coordinates (s, d) back to Cartesian coordinates (x, y),
        compute longitudinal velocity (v_x), and orientation (theta) for one sample.

        Args:
            sample (Sample): A single Sample object containing Frenet trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Cartesian points [[x1, y1], [x2, y2], ...]
            - Corresponding timestamps
            - Longitudinal velocities (v_x) for each time step
            - Orientations (theta) for each time step
        """
        cartesian_points = []
        vx_values = []
        theta_values = []

        # Precompute cumulative lengths of the reference trajectory
        cumulative_lengths = np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
        cumulative_lengths = np.concatenate(([0], cumulative_lengths))

        # Iterate through each time step in the sample
        for s, d, d_d, s_d in zip(sample.s, sample.d, sample.d_d, sample.s_d):
            # Find the closest point on the reference trajectory for the given s
            idx = np.argmin(np.abs(cumulative_lengths - s))
            if idx < len(self.x) - 1:
                base_point = np.array([self.x[idx], self.y[idx]])
            else:
                base_point = np.array([self.x[-1], self.y[-1]])

            # Compute tangent and normal vectors
            dx = np.gradient(self.x)
            dy = np.gradient(self.y)
            tangent = np.array([dx[idx], dy[idx]])
            tangent_unit = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent_unit[1], tangent_unit[0]])

            # Compute Cartesian point
            cartesian_point = base_point + d * normal
            cartesian_points.append(cartesian_point)

            # Lookup curvature and compute longitudinal velocity (v_x)
            kappa_r = self.curv[idx]
            vx = np.sqrt(((1 - kappa_r * d) ** 2) * s_d**2 + d_d**2)
            vx_values.append(vx)

            # Compute orientation (theta)
            theta_c = np.arctan2(tangent_unit[1], tangent_unit[0])
            delta_theta = np.arctan2(d_d, s_d * (1 - kappa_r * d))
            theta = delta_theta + theta_c
            theta_values.append(theta)

        return np.array(cartesian_points), np.array(sample.t), np.array(vx_values), np.array(theta_values)
