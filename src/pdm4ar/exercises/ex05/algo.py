from collections.abc import Sequence
from math import sin, cos, pi, tan

from dg_commons import SE2Transform

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
    min_r = wheel_base / tan(max_steering_angle)
    return DubinsParam(min_radius=min_r)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
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

    d = np.linalg.norm(circle_end.center.p - circle_start.center.p, ord=2)
    radius_2x = circle_end.radius + circle_start.radius

    if d == 0:
        if circle_end.type != circle_start.type:
            return []
        else:
            return [
                Line(
                    start_config=SE2Transform(circle_start.start_config.p, circle_start.start_config.theta),
                    end_config=SE2Transform(circle_start.start_config.p, circle_start.start_config.theta),
                )
            ]

    if radius_2x < d:
        if circle_end.type != circle_start.type:
            return compute_inner_tangent(circle_start, circle_end)
        else:
            return compute_outer_tangent(circle_start, circle_end)

    if radius_2x > d:
        if circle_end.type != circle_start.type:
            return []
        else:
            return compute_outer_tangent(circle_start, circle_end)

    if radius_2x == d:
        if circle_end.type != circle_start.type:
            return compute_zero_tangent(circle_start, circle_end)
        else:
            return compute_outer_tangent(circle_start, circle_end)

    return []


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    path = []
    best_path = []
    best_length = np.inf

    start_circles = calculate_turning_circles(start_config, radius)
    end_circles = calculate_turning_circles(end_config, radius)

    for sc in [start_circles.left, start_circles.right]:
        for ec in [end_circles.left, end_circles.right]:

            path = calculate_tangent_btw_circles(sc, ec)
            if not path:
                continue

            # Compute the arcs that connect to the tangents
            start_arc = compute_first_arc(sc, path[0].start_config, radius)
            end_arc = compute_last_arc(ec, path[0].end_config, radius)

            # Build path
            path = [start_arc] + path + [end_arc]
            length = sum(segment.length for segment in path)

            # # Check optimality
            if length < best_length:
                best_length = length
                best_path = path

    path = compute_arc_tangent(start_circles.left, end_circles.left)
    if path != []:
        start_arc = compute_first_arc(start_circles.left, path[0].start_config, radius)
        end_arc = compute_last_arc(end_circles.left, path[0].end_config, radius)
        path = [start_arc] + path + [end_arc]
        length = sum(segment.length for segment in path)
        if length < best_length:
            best_length = length
            best_path = path

    # RLR path
    path = compute_arc_tangent(start_circles.right, end_circles.right)
    if path != []:
        start_arc = compute_first_arc(start_circles.right, path[0].start_config, radius)
        end_arc = compute_last_arc(end_circles.right, path[0].end_config, radius)
        path = [start_arc] + path + [end_arc]
        length = sum(segment.length for segment in path)
        if length < best_length:
            best_length = length
            best_path = path

    # Please keep segments with zero length in the return list & return a valid dubins path!
    return best_path  # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    path_forward = calculate_dubins_path(start_config, end_config, radius)
    length_forward = sum(segment.length for segment in path_forward)

    path_reverse = calculate_dubins_path(end_config, start_config, radius)
    length_reverse = sum(segment.length for segment in path_reverse)

    if length_forward < length_reverse:
        return path_forward
    else:
        return path_reverser(path_reverse)

    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]


def compute_outer_tangent(circle_start: Curve, circle_end: Curve) -> list[Line]:

    d_vector = circle_end.center.p - circle_start.center.p
    angle = np.arctan2(d_vector[1], d_vector[0])

    offset = circle_start.radius * np.array((sin(angle), -cos(angle)))

    if circle_start.type == DubinsSegmentType.LEFT:
        start = circle_start.center.p + offset
        end = circle_end.center.p + offset
    else:
        start = circle_start.center.p - offset
        end = circle_end.center.p - offset

    return [
        Line(
            start_config=SE2Transform(start, angle),
            end_config=SE2Transform(end, angle),
        )
    ]


def compute_inner_tangent(start_circle: Curve, end_circle: Curve) -> list[Line]:
    d_vector = end_circle.center.p - start_circle.center.p
    d = np.linalg.norm(d_vector)

    angle = np.arctan2(d_vector[1], d_vector[0])
    beta = np.arccos((2 * start_circle.radius) / d)

    if start_circle.type == DubinsSegmentType.LEFT:
        alpha = angle - beta
    else:
        alpha = angle + beta

    offset = start_circle.radius * np.array((cos(alpha), sin(alpha)))

    start = start_circle.center.p + offset
    end = end_circle.center.p - offset

    line_vector = end - start
    line_theta = np.arctan2(line_vector[1], line_vector[0])

    return [
        Line(
            start_config=SE2Transform(start, line_theta),
            end_config=SE2Transform(end, line_theta),
        )
    ]


def compute_first_arc(circle_start: Curve, tangent_pt: SE2Transform, radius: float) -> Curve:

    s_vector = circle_start.start_config.p - circle_start.center.p
    s_angle = np.arctan2(s_vector[1], s_vector[0])

    t_vector = tangent_pt.p - circle_start.center.p
    t_angle = np.arctan2(t_vector[1], t_vector[0])

    if circle_start.type == DubinsSegmentType.LEFT:
        delta = t_angle - s_angle
    else:
        delta = s_angle - t_angle

    if delta < 0:
        delta += 2 * pi
    elif delta > 2 * pi:
        delta -= 2 * pi

    return Curve(
        start_config=circle_start.start_config,
        end_config=tangent_pt,
        center=circle_start.center,
        radius=radius,
        curve_type=circle_start.type,
        arc_angle=delta,
    )


def compute_last_arc(circle_end: Curve, tangent_pt: SE2Transform, radius: float) -> Curve:

    e_vector = circle_end.start_config.p - circle_end.center.p
    e_angle = np.arctan2(e_vector[1], e_vector[0])

    t_vector = tangent_pt.p - circle_end.center.p
    t_angle = np.arctan2(t_vector[1], t_vector[0])

    if circle_end.type == DubinsSegmentType.LEFT:
        delta = e_angle - t_angle
    else:
        delta = t_angle - e_angle

    if delta < 0:
        delta += 2 * pi
    elif delta > 2 * pi:
        delta -= 2 * pi

    return Curve(
        start_config=tangent_pt,
        end_config=circle_end.end_config,
        center=circle_end.center,
        radius=radius,
        curve_type=circle_end.type,
        arc_angle=delta,
    )


def compute_arc_tangent(circle_start: Curve, circle_end: Curve) -> list[Curve]:
    d_vector = circle_end.center.p - circle_start.center.p
    d = np.linalg.norm(d_vector)
    angle = np.arctan2(d_vector[1], d_vector[0])

    if d == 0 or d > 4 * circle_start.radius:
        return []

    try:
        gamma = np.arccos(d / (4 * circle_start.radius))
    except:
        return []

    if circle_start.type == DubinsSegmentType.RIGHT:
        arc_center = circle_start.center.p + 2 * circle_start.radius * np.array(
            [cos(angle - gamma), sin(angle - gamma)]
        )
        arc_start = circle_start.center.p + circle_start.radius * np.array([cos(angle - gamma), sin(angle - gamma)])
    else:
        arc_center = circle_start.center.p + 2 * circle_start.radius * np.array(
            [cos(angle + gamma), sin(angle + gamma)]
        )
        arc_start = circle_start.center.p + circle_start.radius * np.array([cos(angle + gamma), sin(angle + gamma)])

    arc_end = arc_center + circle_start.radius * (circle_end.center.p - arc_center) / np.linalg.norm(
        circle_end.center.p - arc_center
    )

    temp = circle_end.center.p - arc_center
    if circle_start.type == DubinsSegmentType.RIGHT:
        theta_arc_start = -np.pi / 2 + angle - gamma
        theta_arc_end = np.pi / 2 + np.arctan2(temp[1], temp[0])
    else:
        theta_arc_start = np.pi / 2 + angle + gamma
        theta_arc_end = -np.pi / 2 + np.arctan2(temp[1], temp[0])

    s_vector = arc_start - arc_center
    s_angle = np.arctan2(s_vector[1], s_vector[0])

    e_vector = arc_end - arc_center
    e_angle = np.arctan2(e_vector[1], e_vector[0])

    if circle_start.type == DubinsSegmentType.RIGHT:
        delta = e_angle - s_angle
        type = DubinsSegmentType.LEFT
    else:
        delta = s_angle - e_angle
        type = DubinsSegmentType.RIGHT

    if delta < 0:
        delta += 2 * pi
    elif delta > 2 * pi:
        delta -= 2 * pi

    return [
        Curve(
            start_config=SE2Transform(arc_start, theta_arc_start),
            end_config=SE2Transform(arc_end, theta_arc_end),
            center=SE2Transform(arc_center, 0),
            radius=circle_start.radius,
            curve_type=type,
            arc_angle=delta,
        )
    ]


def path_reverser(path: list[Curve | Line]) -> list[Curve | Line]:

    path.reverse()
    for piece in path:
        temp = piece.start_config
        piece.start_config = piece.end_config
        piece.end_config = temp
        piece.gear = Gear.REVERSE

    return path


def compute_zero_tangent(start_circle: Curve, end_circle: Curve) -> list[Line]:
    middle = (end_circle.center.p + start_circle.center.p) / 2
    d_vector = end_circle.center.p - start_circle.center.p
    d = np.linalg.norm(d_vector)

    angle = np.arctan2(d_vector[1], d_vector[0])

    if start_circle.type == DubinsSegmentType.LEFT:
        angle += np.pi / 2
    else:
        angle -= np.pi / 2

    return [
        Line(
            start_config=SE2Transform(middle, angle),
            end_config=SE2Transform(middle, angle),
        )
    ]
