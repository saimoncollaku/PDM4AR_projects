from typing import Tuple
import numpy as np
from scipy.interpolate import splprep, splev

from pdm4ar.exercises.ex12.sampler.sample import Sample


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

    def get_xy(self, coordinate: list[tuple[float, float]]):
        """
        From fresnet coordinate [(s, d)...] to [(x, y)...]
        """
        cartesian_points = np.zeros((len(coordinate), 2))

        # Precompute cumulative lengths of the reference trajectory
        cumulative_lengths = np.cumsum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2))
        cumulative_lengths = np.concatenate(([0], cumulative_lengths))

        # Compute gradients for the reference trajectory
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        for i, (s, d) in enumerate(coordinate):
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

    def to_cartesian(self, sample: Sample):
        """
        Convert Frenet coordinates (s, d) back to Cartesian coordinates (x, y),
        and compute longitudinal velocity (v_x), orientation (theta), and curvature (kappa).

        Args:
            sample (Sample): A single Sample object containing Frenet trajectory.
        Changes in place
        """

        cartesian_points = self.get_xy(list(zip(sample.s, sample.d)))
        sample.x = cartesian_points[:, 0]
        sample.y = cartesian_points[:, 1]
        sample.store_vx_theta()
