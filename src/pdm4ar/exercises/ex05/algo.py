from collections.abc import Sequence
from math import sin, cos, pi, tan

from dg_commons import SE2Transform
from numpy import arctan

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


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
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    min_r = wheel_base * tan(pi / 2 - max_steering_angle)
    return DubinsParam(min_radius=min_r)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    c_r = SE2Transform.identity()
    c_r.p = np.array(
        [
            radius * sin(current_config.theta) + current_config.p[0],
            -radius * cos(current_config.theta) + current_config.p[1],
        ]
    )

    c_l = SE2Transform.identity()
    c_l.p = np.array(
        [
            -radius * sin(current_config.theta) + current_config.p[0],
            +radius * cos(current_config.theta) + current_config.p[1],
        ]
    )

    right_circle = Curve.create_circle(
        center=c_r,
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.RIGHT,
    )

    left_circle = Curve.create_circle(
        center=c_l,
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.LEFT,
    )
    return TurningCircle(left=left_circle, right=right_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
    # TODO implement here your solution
    return []  # i.e., [Line(),...]


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    return []  # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    return []  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]
