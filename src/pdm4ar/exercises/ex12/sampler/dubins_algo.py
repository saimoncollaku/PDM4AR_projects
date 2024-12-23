from collections.abc import Sequence
from enum import Enum
from math import floor
from typing import Optional, Union
import unittest

from typing import List
from dg_commons import SE2Transform
import numpy as np
import matplotlib.pyplot as plt

from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from pdm4ar.exercises.ex12.sampler.dubins_structures import (
    DubinsParam,
    ABC,
    Gear,
    abstractmethod,
    TurningCircle,
    Curve,
    DubinsSegmentType,
    Line,
    Segment,
    Path,
    mod_2_pi,
)


class WrongRadiusError(ValueError):
    pass


class TooMuchSteeringError(ValueError):
    pass


class WrongDimensionError(ValueError):
    pass


class BeyondListError(IndexError):
    pass


class Dubins:
    def __init__(self, wheel_base: float, max_steering_angle: float):
        self.min_radius = self.calculate_car_turning_radius(wheel_base, max_steering_angle)
        assert self.min_radius > 0, "Minimum radius has to be larger than 0"

    def compute_path(self, start: SE2Transform, end: SE2Transform, step_distance: float) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        start_circle, mid_segment, end_circle = self.calculate_dubins_path(
            start_config=start, end_config=end, radius=self.min_radius
        )
        self.c1, self.c2, self.c3 = start_circle, mid_segment, end_circle

        se2_list = [start]
        start_list, left_distance = self.extract_path_points(start_circle, step_distance, step_distance)

        se2_list.extend(start_list)
        if mid_segment is not None:
            mid_list, left_distance = self.extract_path_points(mid_segment, step_distance, left_distance)
            se2_list.extend(mid_list)
        end_list, left_distance = self.extract_path_points(end_circle, step_distance, left_distance)
        se2_list.extend(end_list)
        if not np.isclose(left_distance, 0):
            se2_list.append(end)
        return se2_list

    def calculate_car_turning_radius(self, wheel_base: float, max_steering_angle: float) -> float:
        min_radius = wheel_base / np.tan(max_steering_angle)
        return min_radius

    def calculate_turning_circles(self, current_config: SE2Transform, radius: float) -> TurningCircle:
        t_ab = current_config.as_SE2()
        t_bl = SE2Transform([0, radius], 0).as_SE2()
        t_br = SE2Transform([0, -radius], 0).as_SE2()
        t_al: np.ndarray = t_ab @ t_bl
        t_ar: np.ndarray = t_ab @ t_br

        # index
        left_circle = Curve.create_circle(
            center=SE2Transform(t_al[0:2, 2].tolist(), 0),
            config_on_circle=current_config,
            radius=radius,
            curve_type=DubinsSegmentType.LEFT,
        )
        # index
        right_circle = Curve.create_circle(
            center=SE2Transform(t_ar[0:2, 2].tolist(), 0),
            config_on_circle=current_config,
            radius=radius,
            curve_type=DubinsSegmentType.RIGHT,
        )
        return TurningCircle(left=left_circle, right=right_circle)

    def calculate_tangent_btw_circles(self, circle_start: Curve, circle_end: Curve) -> list[Line]:
        c1 = circle_start.center.p
        c2 = circle_end.center.p
        if not circle_start.radius == circle_end.radius:
            raise WrongRadiusError("Radii not equal")
        r = circle_end.radius
        center_vector = c2 - c1
        if not len(center_vector) == 2:
            raise WrongDimensionError("Center Vector is wrong dimension")
        center_distances = np.linalg.norm(center_vector)
        # index
        center_theta = np.arctan2(center_vector[1], center_vector[0])

        cs_config = circle_start.type
        ce_config = circle_end.type
        if cs_config == DubinsSegmentType.LEFT:
            if ce_config == DubinsSegmentType.LEFT:
                # outer tangent
                t_i_tangent1 = c1 + np.array([r * np.sin(center_theta), -r * np.cos(center_theta)])
                t_i_tangent2 = c2 + np.array([r * np.sin(center_theta), -r * np.cos(center_theta)])
                start = SE2Transform(t_i_tangent1.tolist(), center_theta)
                end = SE2Transform(t_i_tangent2.tolist(), center_theta)
                return [Line(start_config=start, end_config=end)]
            elif ce_config == DubinsSegmentType.RIGHT:
                # inner tangent
                if center_distances < 2 * r:
                    return []
                angle = center_theta + np.arcsin(2 * r / center_distances)
                t_i_tangent1 = c1 + np.array([r * np.sin(angle), -r * np.cos(angle)])
                t_i_tangent2 = c2 + np.array([-r * np.sin(angle), r * np.cos(angle)])
                start = SE2Transform(t_i_tangent1.tolist(), angle)
                end = SE2Transform(t_i_tangent2.tolist(), angle)
                return [Line(start_config=start, end_config=end)]
        elif cs_config == DubinsSegmentType.RIGHT:
            if ce_config == DubinsSegmentType.LEFT:
                # inner tangent
                if center_distances < 2 * r:
                    return []
                angle = center_theta - np.arcsin(2 * r / center_distances)
                t_i_tangent1 = c1 + np.array([-r * np.sin(angle), r * np.cos(angle)])
                t_i_tangent2 = c2 + np.array([r * np.sin(angle), -r * np.cos(angle)])
                start = SE2Transform(t_i_tangent1.tolist(), angle)
                end = SE2Transform(t_i_tangent2.tolist(), angle)
                return [Line(start_config=start, end_config=end)]
            elif ce_config == DubinsSegmentType.RIGHT:
                # outer tangent
                t_i_tangent1 = c1 + np.array([-r * np.sin(center_theta), r * np.cos(center_theta)])
                t_i_tangent2 = c2 + np.array([-r * np.sin(center_theta), r * np.cos(center_theta)])
                start = SE2Transform(t_i_tangent1.tolist(), center_theta)
                end = SE2Transform(t_i_tangent2.tolist(), center_theta)
                return [Line(start_config=start, end_config=end)]
        raise ValueError(f"Wrong configuration of circles found {cs_config}, {ce_config}")

    def third_circle_curve_above(self, circle_start: Curve, circle_end: Curve) -> Optional[Curve]:
        c1 = circle_start.center.p
        c2 = circle_end.center.p
        r = circle_end.radius
        center_vector = c2 - c1
        center_distances = np.linalg.norm(center_vector)
        if 4 * r < center_distances:
            return None
        # index
        center_theta = np.arctan2(center_vector[1], center_vector[0])

        cs_config = circle_start.type
        ce_config = circle_end.type
        if cs_config != ce_config:
            return None

        delta = np.arccos(center_distances / (2 * 2 * r))
        center = c1 + np.array([2 * r * np.cos(center_theta + delta), 2 * r * np.sin(center_theta + delta)])
        start = c1 + np.array([r * np.cos(center_theta + delta), r * np.sin(center_theta + delta)])
        end = c2 + np.array(
            [-r * np.cos(center_theta - delta), -r * np.sin(center_theta - delta)]
        )  # center_theta + (pi - delta)
        if cs_config == DubinsSegmentType.RIGHT:
            start_config = SE2Transform(start.tolist(), center_theta - (np.pi / 2 - delta))
            end_config = SE2Transform(end.tolist(), center_theta + (np.pi / 2 - delta))
            direction = DubinsSegmentType.LEFT
        else:
            # anomaly case with longer route
            start_config = SE2Transform(start.tolist(), center_theta - (np.pi / 2 - delta) + np.pi)
            end_config = SE2Transform(end.tolist(), center_theta + (np.pi / 2 - delta) + np.pi)
            direction = DubinsSegmentType.RIGHT
        center_config = SE2Transform(center.tolist(), 0)
        curve = Curve(start_config, end_config, center_config, r, direction, 0)
        return curve

    def third_circle_curve_below(self, circle_start: Curve, circle_end: Curve) -> Optional[Curve]:
        c1 = circle_start.center.p
        c2 = circle_end.center.p
        r = circle_end.radius
        center_vector = c2 - c1
        center_distances = np.linalg.norm(center_vector)
        if 4 * r < center_distances:
            return None
        # index
        center_theta = np.arctan2(center_vector[1], center_vector[0])

        cs_config = circle_start.type
        ce_config = circle_end.type
        if cs_config != ce_config:
            return None

        delta = np.arccos(center_distances / (2 * 2 * r))
        start = c1 + np.array([r * np.cos(center_theta - delta), r * np.sin(center_theta - delta)])
        end = c2 + np.array(
            [-r * np.cos(center_theta + delta), -r * np.sin(center_theta + delta)]
        )  # center_theta - (pi - delta)
        center = c1 + np.array([2 * r * np.cos(center_theta - delta), 2 * r * np.sin(center_theta - delta)])
        if cs_config == DubinsSegmentType.LEFT:
            start_config = SE2Transform(start.tolist(), center_theta + (np.pi / 2 - delta))
            end_config = SE2Transform(end.tolist(), center_theta - (np.pi / 2 - delta))
            direction = DubinsSegmentType.RIGHT
        else:
            # anomaly case with longer route
            start_config = SE2Transform(start.tolist(), center_theta + (np.pi / 2 - delta) + np.pi)
            end_config = SE2Transform(end.tolist(), center_theta - (np.pi / 2 - delta) + np.pi)
            direction = DubinsSegmentType.LEFT
        center_config = SE2Transform(center.tolist(), 0)
        curve = Curve(start_config, end_config, center_config, r, direction, 0)
        return curve

    def change_arc_angle_between(self, circle: Curve) -> float:
        """
        Changes internal arc angle and length of arc
        based on the start and end configuration set on the circle
        """
        diff = circle.end_config.theta - circle.start_config.theta
        if circle.type == DubinsSegmentType.LEFT:
            if diff < 0:
                diff = 2 * np.pi + diff
            else:
                pass
        elif circle.type == DubinsSegmentType.RIGHT:
            if diff < 0:
                diff = -diff
            else:
                diff = 2 * np.pi - diff
        circle.arc_angle = diff
        return circle.arc_angle

    def csc_calculate(self, circle1: Curve, circle2: Curve):
        lines = self.calculate_tangent_btw_circles(circle1, circle2)
        if len(lines) == 0:
            return 0.0, None
        line = lines[0]
        circle1.end_config = line.start_config
        circle2.start_config = line.end_config
        self.change_arc_angle_between(circle1)
        self.change_arc_angle_between(circle2)
        return circle1.length + line.length + circle2.length, line

    def ccc_above_calculate(self, circle1: Curve, circle2: Curve):
        mid_circle = self.third_circle_curve_above(circle1, circle2)
        if mid_circle is None:
            return 0.0, None
        circle1.end_config = mid_circle.start_config
        circle2.start_config = mid_circle.end_config
        self.change_arc_angle_between(circle1)
        self.change_arc_angle_between(circle2)
        self.change_arc_angle_between(mid_circle)
        return circle1.length + mid_circle.length + circle2.length, mid_circle

    def ccc_below_calculate(self, circle1: Curve, circle2: Curve):
        mid_circle = self.third_circle_curve_below(circle1, circle2)
        if mid_circle is None:
            return 0.0, None
        circle1.end_config = mid_circle.start_config
        circle2.start_config = mid_circle.end_config
        self.change_arc_angle_between(circle1)
        self.change_arc_angle_between(circle2)
        self.change_arc_angle_between(mid_circle)
        return circle1.length + mid_circle.length + circle2.length, mid_circle

    def calculate_dubins_path(
        self, start_config: SE2Transform, end_config: SE2Transform, radius: float
    ) -> tuple[Curve, Optional[Segment], Curve]:
        # Have to go through all possible Dubins path and check their lengths
        start_circles = self.calculate_turning_circles(start_config, radius)
        end_circles = self.calculate_turning_circles(end_config, radius)
        sr_circle = start_circles.right
        sl_circle = start_circles.left
        er_circle = end_circles.right
        el_circle = end_circles.left

        # index
        total_lengths = [-1.0] * 8  # 8 configurations

        # RSR
        total_len, curve = self.csc_calculate(sr_circle, er_circle)
        total_lengths[0] = total_len if curve else -1.0

        # LSL
        total_len, curve = self.csc_calculate(sl_circle, el_circle)
        total_lengths[1] = total_len if curve else -1.0

        # RSL
        total_len, curve = self.csc_calculate(sr_circle, el_circle)
        total_lengths[2] = total_len if curve else -1.0
        # LSR
        total_len, curve = self.csc_calculate(sl_circle, er_circle)
        total_lengths[3] = total_len if curve else -1.0

        # RLR
        total_len, curve = self.ccc_above_calculate(sr_circle, er_circle)
        total_lengths[4] = total_len if curve else -1.0

        # LRL
        total_len, curve = self.ccc_above_calculate(sl_circle, el_circle)
        total_lengths[5] = total_len if curve else -1.0

        # RLR
        total_len, curve = self.ccc_below_calculate(sr_circle, er_circle)
        total_lengths[6] = total_len if curve else -1.0

        # LRL
        total_len, curve = self.ccc_below_calculate(sl_circle, el_circle)
        total_lengths[7] = total_len if curve else -1.0

        # index
        min_idx = -1
        min_length = max(total_lengths)
        for i, tl in enumerate(total_lengths):
            if tl >= 0 and min_length >= tl:
                min_idx = i
                min_length = tl

        # print(total_lengths)

        paths = []
        if min_idx == 0:
            total_len, line = self.csc_calculate(sr_circle, er_circle)
            paths = (sr_circle, line, er_circle)
        elif min_idx == 1:
            total_len, line = self.csc_calculate(sl_circle, el_circle)
            paths = (sl_circle, line, el_circle)
        elif min_idx == 2:
            total_len, line = self.csc_calculate(sr_circle, el_circle)
            paths = (sr_circle, line, el_circle)
        elif min_idx == 3:
            total_len, line = self.csc_calculate(sl_circle, er_circle)
            paths = (sl_circle, line, er_circle)
        elif min_idx == 4:
            total_len, mid_circle = self.ccc_above_calculate(sr_circle, er_circle)
            paths = (sr_circle, mid_circle, er_circle)
        elif min_idx == 5:
            total_len, mid_circle = self.ccc_above_calculate(sl_circle, el_circle)
            paths = (sl_circle, mid_circle, el_circle)
        elif min_idx == 6:
            total_len, mid_circle = self.ccc_below_calculate(sr_circle, er_circle)
            paths = (sr_circle, mid_circle, er_circle)
        elif min_idx == 7:
            total_len, mid_circle = self.ccc_below_calculate(sl_circle, el_circle)
            paths = (sl_circle, mid_circle, el_circle)
        else:
            raise BeyondListError(f"Invalid min_idx {min_idx}")

        return paths

    def get_rot_matrix(self, alpha: float) -> np.ndarray:
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return rot_matrix

    def get_next_point_on_curve(self, curve: Curve, point: SE2Transform, delta_angle: float) -> SE2Transform:
        point_translated = point.p - curve.center.p
        rot_matrix = self.get_rot_matrix(delta_angle)
        next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)
        return next_point

    def get_next_point_on_line(self, line: Line, point: SE2Transform, delta_length: float) -> SE2Transform:
        return SE2Transform(point.p + delta_length * line.direction, theta=point.theta)

    def interpolate_line_points(
        self, line: Line, start_length: float, step_length: float
    ) -> tuple[List[SE2Transform], float]:
        pts_list = []
        old_point = self.get_next_point_on_line(line, line.start_config, start_length)

        total_length = line.length - start_length
        num_samples = floor(total_length / step_length)
        for _ in range(num_samples):
            pts_list.append(old_point)
            point_next = self.get_next_point_on_line(line, old_point, step_length)
            old_point = point_next
        left_distance = total_length - num_samples * step_length
        return pts_list, left_distance

    def interpolate_curve_points(
        self, curve: Curve, start_angle: float, step_angle: float
    ) -> tuple[List[SE2Transform], float]:
        pts_list = []
        old_point = self.get_next_point_on_curve(curve, curve.start_config, delta_angle=start_angle)

        angle = curve.arc_angle - start_angle
        direction = curve.type
        angle = curve.gear.value * direction.value * angle
        num_samples = floor(angle / step_angle)
        for _ in range(num_samples):
            pts_list.append(old_point)
            point_next = self.get_next_point_on_curve(curve, point=old_point, delta_angle=step_angle)
            old_point = point_next
        left_angle = angle - num_samples * step_angle
        return pts_list, left_angle

    def extract_path_points(
        self, seg: Segment, max_distance_per_step: float, start_distance: float
    ) -> tuple[List[SE2Transform], float]:
        """Extracts a fixed number of SE2Transform points on a path"""
        seg.start_config.theta = mod_2_pi(seg.start_config.theta)
        seg.end_config.theta = mod_2_pi(seg.end_config.theta)
        if seg.type is DubinsSegmentType.STRAIGHT:
            assert isinstance(seg, Line)
            line_pts, left_distance = self.interpolate_line_points(seg, start_distance, max_distance_per_step)
            return line_pts, left_distance
        else:  # Curve
            assert isinstance(seg, Curve)
            init_angle = start_distance / self.min_radius
            step_angle = max_distance_per_step / self.min_radius
            curve_pts, left_angle = self.interpolate_curve_points(seg, init_angle, step_angle)
            left_distance = self.min_radius * left_angle
            return curve_pts, left_distance

    def plot_trajectory(self, trajectory: list[SE2Transform], start_config: SE2Transform, end_config: SE2Transform):
        x = [state.p[0] for state in trajectory]
        y = [state.p[1] for state in trajectory]
        psi = [state.theta for state in trajectory]
        start_state = [start_config.p[0], start_config.p[1], start_config.theta]
        end_state = [end_config.p[0], end_config.p[1], end_config.theta]

        # Calculate the components of the arrow (unit vectors scaled for visualization)
        arrow_length = 0.2  # Length of the arrows
        u = [arrow_length * np.cos(angle) for angle in psi]  # X-component
        v = [arrow_length * np.sin(angle) for angle in psi]  # Y-component

        # Plot the states and heading
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, "bo-", label="Path")  # Path
        plt.quiver(
            x, y, u, v, angles="xy", scale_units="xy", scale=1, color="red", label="Heading"
        )  # Arrows for heading

        # Plot the starting state
        start_arrow_u = arrow_length * np.cos(start_state[2])  # X-component of the starting arrow
        start_arrow_v = arrow_length * np.sin(start_state[2])  # Y-component of the starting arrow
        plt.scatter(start_state[0], start_state[1], color="green", s=100, label="Start", marker="*")  # Start marker
        plt.quiver(
            start_state[0],
            start_state[1],
            start_arrow_u,
            start_arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
        )

        # Plot the ending state
        end_arrow_u = arrow_length * np.cos(end_state[2])  # X-component of the ending arrow
        end_arrow_v = arrow_length * np.sin(end_state[2])  # Y-component of the ending arrow
        plt.scatter(end_state[0], end_state[1], color="orange", s=100, label="End", marker="s")  # End marker
        plt.quiver(
            end_state[0], end_state[1], end_arrow_u, end_arrow_v, angles="xy", scale_units="xy", scale=1, color="orange"
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Path with Heading Angles")
        plt.legend()
        plt.axis("equal")  # Ensure equal scaling for x and y axes
        plt.grid()
        plt.savefig("dubins.png")
        plt.close()


class DubinsTest(unittest.TestCase):
    dubins: Dubins

    def setUp(self):
        wheel_base = 1.2
        max_acceleration = 5
        max_steering_angle = 1
        self.dubins = Dubins(wheel_base, max_steering_angle)
        v_max = np.sqrt(max_acceleration * wheel_base / np.tan(max_steering_angle))
        dt = 0.1
        self.step_length = v_max * dt

    def testcase1(self):
        start_config = SE2Transform([-26.27269412470485, 8.787346780714294], -0.011495208045596282)
        end_config = SE2Transform([-27.132346369790178, 6.537920517155902], 0.0032153572)
        trajectory = self.dubins.compute_path(start_config, end_config, self.step_length)
        self.dubins.plot_trajectory(trajectory, start_config, end_config)


if __name__ == "__main__":
    unittest.main()
