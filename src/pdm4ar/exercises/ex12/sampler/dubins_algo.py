from collections.abc import Sequence
from enum import Enum
from typing import Optional, Union

from typing import List
from dg_commons import SE2Transform
import numpy as np

from pdm4ar.exercises.ex12.sampler.dubins_structures import (
    DubinsParam,
    ABC,
    Gear,
    abstractmethod,
    TurningCircle,
    Curve,
    DubinsSegmentType,
    Line,
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


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = self.calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = self.extract_path_points(path)
        return se2_list

    def calculate_car_turning_radius(self, wheel_base: float, max_steering_angle: float) -> DubinsParam:
        min_radius = wheel_base / np.tan(max_steering_angle)
        return DubinsParam(min_radius=min_radius)

    @staticmethod
    def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
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

    @staticmethod
    def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
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

    @staticmethod
    def third_circle_curve_above(circle_start: Curve, circle_end: Curve) -> Optional[Curve]:
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

    @staticmethod
    def third_circle_curve_below(circle_start: Curve, circle_end: Curve) -> Optional[Curve]:
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

    @staticmethod
    def change_arc_angle_between(circle: Curve) -> float:
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

    @staticmethod
    def csc_calculate(circle1: Curve, circle2: Curve):
        lines = Dubins.calculate_tangent_btw_circles(circle1, circle2)
        if len(lines) == 0:
            return 0.0, None
        line = lines[0]
        circle1.end_config = line.start_config
        circle2.start_config = line.end_config
        Dubins.change_arc_angle_between(circle1)
        Dubins.change_arc_angle_between(circle2)
        return circle1.length + line.length + circle2.length, line

    @staticmethod
    def ccc_above_calculate(circle1: Curve, circle2: Curve):
        mid_circle = Dubins.third_circle_curve_above(circle1, circle2)
        if mid_circle is None:
            return 0.0, None
        circle1.end_config = mid_circle.start_config
        circle2.start_config = mid_circle.end_config
        Dubins.change_arc_angle_between(circle1)
        Dubins.change_arc_angle_between(circle2)
        Dubins.change_arc_angle_between(mid_circle)
        return circle1.length + mid_circle.length + circle2.length, mid_circle

    @staticmethod
    def ccc_below_calculate(circle1: Curve, circle2: Curve):
        mid_circle = Dubins.third_circle_curve_below(circle1, circle2)
        if mid_circle is None:
            return 0.0, None
        circle1.end_config = mid_circle.start_config
        circle2.start_config = mid_circle.end_config
        Dubins.change_arc_angle_between(circle1)
        Dubins.change_arc_angle_between(circle2)
        Dubins.change_arc_angle_between(mid_circle)
        return circle1.length + mid_circle.length + circle2.length, mid_circle

    @staticmethod
    def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
        # Have to go through all possible Dubins path and check their lengths
        start_circles = Dubins.calculate_turning_circles(start_config, radius)
        end_circles = Dubins.calculate_turning_circles(end_config, radius)
        sr_circle = start_circles.right
        sl_circle = start_circles.left
        er_circle = end_circles.right
        el_circle = end_circles.left

        # index
        total_lengths = [-1.0] * 8  # 8 configurations

        # RSR
        total_len, curve = Dubins.csc_calculate(sr_circle, er_circle)
        total_lengths[0] = total_len if curve else -1.0

        # LSL
        total_len, curve = Dubins.csc_calculate(sl_circle, el_circle)
        total_lengths[1] = total_len if curve else -1.0

        # RSL
        total_len, curve = Dubins.csc_calculate(sr_circle, el_circle)
        total_lengths[2] = total_len if curve else -1.0
        # LSR
        total_len, curve = Dubins.csc_calculate(sl_circle, er_circle)
        total_lengths[3] = total_len if curve else -1.0

        # RLR
        total_len, curve = Dubins.ccc_above_calculate(sr_circle, er_circle)
        total_lengths[4] = total_len if curve else -1.0

        # LRL
        total_len, curve = Dubins.ccc_above_calculate(sl_circle, el_circle)
        total_lengths[5] = total_len if curve else -1.0

        # RLR
        total_len, curve = Dubins.ccc_below_calculate(sr_circle, er_circle)
        total_lengths[6] = total_len if curve else -1.0

        # LRL
        total_len, curve = Dubins.ccc_below_calculate(sl_circle, el_circle)
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
            total_len, line = Dubins.csc_calculate(sr_circle, er_circle)
            paths = [sr_circle, line, er_circle]
        elif min_idx == 1:
            total_len, line = Dubins.csc_calculate(sl_circle, el_circle)
            paths = [sl_circle, line, el_circle]
        elif min_idx == 2:
            total_len, line = Dubins.csc_calculate(sr_circle, el_circle)
            paths = [sr_circle, line, el_circle]
        elif min_idx == 3:
            total_len, line = Dubins.csc_calculate(sl_circle, er_circle)
            paths = [sl_circle, line, er_circle]
        elif min_idx == 4:
            total_len, mid_circle = Dubins.ccc_above_calculate(sr_circle, er_circle)
            paths = [sr_circle, mid_circle, er_circle]
        elif min_idx == 5:
            total_len, mid_circle = Dubins.ccc_above_calculate(sl_circle, el_circle)
            paths = [sl_circle, mid_circle, el_circle]
        elif min_idx == 6:
            total_len, mid_circle = Dubins.ccc_below_calculate(sr_circle, er_circle)
            paths = [sr_circle, mid_circle, er_circle]
        elif min_idx == 7:
            total_len, mid_circle = Dubins.ccc_below_calculate(sl_circle, el_circle)
            paths = [sl_circle, mid_circle, el_circle]
        else:
            raise BeyondListError("Invalid min_idx %d", min_idx)

        return paths

    @staticmethod
    def get_rot_matrix(alpha: float) -> np.ndarray:
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return rot_matrix

    @staticmethod
    def get_next_point_on_curve(curve: Curve, point: SE2Transform, delta_angle: float) -> SE2Transform:
        point_translated = point.p - curve.center.p
        rot_matrix = Dubins.get_rot_matrix(delta_angle)
        next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)
        return next_point

    @staticmethod
    def interpolate_line_points(line: Line, number_of_points: int) -> List[SE2Transform]:
        start = line.start_config
        end = line.end_config
        start_to_end = end.p - start.p
        intervals = np.linspace(0, 1.0, number_of_points)
        return [SE2Transform(start.p + i * start_to_end, start.theta) for i in intervals]

    @staticmethod
    def interpolate_curve_points(curve: Curve, number_of_points: int) -> List[SE2Transform]:
        pts_list = []
        angle = curve.arc_angle
        direction = curve.type
        angle = curve.gear.value * direction.value * angle
        split_angle = angle / number_of_points
        old_point = curve.start_config
        for i in range(number_of_points):
            pts_list.append(old_point)
            point_next = Dubins.get_next_point_on_curve(curve, point=old_point, delta_angle=split_angle)
            old_point = point_next
        return pts_list

    @staticmethod
    def extract_path_points(path: Path) -> List[SE2Transform]:
        """Extracts a fixed number of SE2Transform points on a path"""
        pts_list = []
        num_points_per_segment = 20
        for idx, seg in enumerate(path):
            seg.start_config.theta = mod_2_pi(seg.start_config.theta)
            seg.end_config.theta = mod_2_pi(seg.end_config.theta)
            if seg.type is DubinsSegmentType.STRAIGHT:
                assert isinstance(seg, Line)
                line_pts = Dubins.interpolate_line_points(seg, num_points_per_segment)
                pts_list.extend(line_pts)
            else:  # Curve
                assert isinstance(seg, Curve)
                curve_pts = Dubins.interpolate_curve_points(seg, num_points_per_segment)
                pts_list.extend(curve_pts)
        pts_list.append(path[-1].end_config)
        return pts_list
