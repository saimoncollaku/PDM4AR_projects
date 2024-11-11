from http.client import CONTINUE
from math import ceil, floor, isclose
from typing import List
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
    CollisionPrimitives_SeparateAxis,
)
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import shapely
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Set
from dataclasses import dataclass

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert type(p_2) in COLLISION_PRIMITIVES[type(p_1)], "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not


def geo_primitive_to_shapely(p: GeoPrimitive):
    """
    Given function.

    Casts a geometric primitive into a Shapely object. Feel free to use this function or not
    for the later tasks.
    """
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else:  # Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)


def translate_segment(s: Segment, d: float) -> list[Segment]:
    # Segment endpoints as arrays
    p1 = np.array([s.p1.x, s.p1.y], dtype=float)
    p2 = np.array([s.p2.x, s.p2.y], dtype=float)
    direction = p2 - p1

    normal = np.array([direction[1], -direction[0]])
    normal /= np.linalg.norm(normal)
    t = normal * d

    p1_up = p1 + t
    p2_up = p2 + t
    p1_dw = p1 - t
    p2_dw = p2 - t

    return [
        Segment(Point(p1_up[0], p1_up[1]), Point(p2_up[0], p2_up[1])),
        Segment(Point(p1_dw[0], p1_dw[1]), Point(p2_dw[0], p2_dw[1])),
    ]


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        processed_obstacles = []
        for obstacle in obstacles:
            if isinstance(obstacle, Triangle):
                processed_obstacles.append(Polygon([obstacle.v1, obstacle.v2, obstacle.v3]))
            else:
                processed_obstacles.append(obstacle)

        circles = [Circle(wp, r) for wp in t.waypoints]

        collision_idx = []

        for idx, step in enumerate(t.waypoints):
            if idx in collision_idx:
                continue
            for o in processed_obstacles:
                if isinstance(o, Circle):
                    d = np.linalg.norm(np.array([step.x - o.center.x, step.y - o.center.y]))
                    if d < r + o.radius:
                        if idx == len(t.waypoints) - 1:
                            collision_idx.append(idx - 1)
                            break
                        if idx != 0:
                            collision_idx.append(idx - 1)
                        collision_idx.append(idx)
                        break
                else:
                    if CollisionPrimitives_SeparateAxis.separating_axis_thm(o, circles[idx])[0]:
                        if idx == len(t.waypoints) - 1:
                            collision_idx.append(idx - 1)
                            break
                        if idx != 0:
                            collision_idx.append(idx - 1)
                        collision_idx.append(idx)
                        break
            if idx in collision_idx:
                continue
            if idx == len(t.waypoints) - 1:
                continue
            s = Segment(t.waypoints[idx], t.waypoints[idx + 1])
            s_t = translate_segment(s, r)
            for o in processed_obstacles:
                if check_collision(s_t[0], o):
                    collision_idx.append(idx)
                    break
                if check_collision(s_t[1], o):
                    collision_idx.append(idx)
                    break

        return collision_idx

    def path_collision_check_occupancy_grid(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        obstacles_shapely = [geo_primitive_to_shapely(o) for o in obstacles]

        # Calculate bounds
        bounds = shapely.total_bounds(obstacles_shapely)
        x_min, y_min, x_max, y_max = bounds.tolist()

        grid = OccupancyGrid(x_min - r, y_min - r, x_max + r, y_max + r, 50, 50)
        grid.update_occupancy(obstacles_shapely)

        circles = [geo_primitive_to_shapely(Circle(wp, r)) for wp in t.waypoints]

        collision_idx = []
        waypoints_count = len(t.waypoints)

        # Process waypoints
        for idx in range(waypoints_count):
            # Check start
            if grid.check_polygon_collision(circles[idx]):
                if idx == waypoints_count - 1:
                    collision_idx.append(idx - 1)
                    continue
                if idx > 0:
                    collision_idx.append(idx - 1)
                collision_idx.append(idx)
                continue

            if idx == waypoints_count - 1:
                continue

            # Check road
            s = Segment(t.waypoints[idx], t.waypoints[idx + 1])
            s_t = translate_segment(s, r)
            path_rect = shapely.geometry.Polygon(
                (
                    (s_t[0].p1.x, s_t[0].p1.y),
                    (s_t[1].p1.x, s_t[1].p1.y),
                    (s_t[1].p2.x, s_t[1].p2.y),
                    (s_t[0].p2.x, s_t[0].p2.y),
                )
            )
            if grid.check_polygon_collision(path_rect):
                collision_idx.append(idx)

        return collision_idx

    def path_collision_check_r_tree(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        obstacles_shapely = []
        collision_idx = set()

        for o in obstacles:
            obstacles_shapely.append(geo_primitive_to_shapely(o))
        tree = shapely.STRtree(obstacles_shapely)

        for idx, step in enumerate(t.waypoints):
            if idx in collision_idx:
                continue
            start = geo_primitive_to_shapely(Circle(step, r))
            potential_collisions = tree.query(start)
            for i in potential_collisions:
                if start.intersects(obstacles_shapely[i]):
                    if idx == len(t.waypoints) - 1:
                        collision_idx.add(idx - 1)
                        break
                    if idx != 0:
                        collision_idx.add(idx - 1)
                    collision_idx.add(idx)
                    break
            if idx in collision_idx:
                continue
            if idx == len(t.waypoints) - 1:
                continue

            s = Segment(t.waypoints[idx], t.waypoints[idx + 1])
            s_t = translate_segment(s, r)
            seg = geo_primitive_to_shapely(s_t[0])
            potential_collisions = tree.query(seg, predicate="intersects")
            for i in potential_collisions:
                if seg.intersects(obstacles_shapely[i]):
                    collision_idx.add(idx)
                    break
            if idx in collision_idx:
                continue
            seg = geo_primitive_to_shapely(s_t[1])
            potential_collisions = tree.query(seg, predicate="intersects")
            for i in potential_collisions:
                if seg.intersects(obstacles_shapely[i]):
                    collision_idx.add(idx)
                    break

        return list(collision_idx)

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: list[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        p1 = current_pose.p
        p2 = next_pose.p
        dist = np.linalg.norm(p1 - p2)
        start = Point(0, 0)
        end = Point(dist, 0)
        path_seg = Path([start, end])

        # Used the faster algo i had
        if self.path_collision_check_r_tree(path_seg, r, observed_obstacles):
            return True
        return False

    def path_collision_check_safety_certificate(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        def get_interpolated_point(start: Point, end: Point, dist: float, total_length: float) -> Point:
            """
            Calculates a new point along the segment using vectorized interpolation.

            Parameters:
                start (Point): Starting point of segment
                end (Point): End point of segment
                dist (float): Distance from start point
                total_length (float): Total segment length

            Returns:
                Point: Interpolated point along segment
            """
            # edge cases
            if isclose(total_length, 0):
                return Point(start.x, start.y)
            if dist >= total_length:
                return Point(end.x, end.y)
            if dist <= 0:
                return Point(start.x, start.y)

            # interpolation ratio
            ratio = dist / total_length
            coords = np.array([[start.x, start.y], [end.x, end.y]])
            new_coords = (1 - ratio) * coords[0] + ratio * coords[1]
            return Point(new_coords[0], new_coords[1])

        obstacles_shapely = [geo_primitive_to_shapely(obstacle) for obstacle in obstacles]
        collision_idx = []

        segment_lengths = []
        for i in range(len(t.waypoints) - 1):
            start = geo_primitive_to_shapely(t.waypoints[i])
            end = geo_primitive_to_shapely(t.waypoints[i + 1])
            segment_lengths.append(start.distance(end))

        for i in range(len(t.waypoints) - 1):
            segment_length = segment_lengths[i]

            if isclose(segment_length, 0):
                continue

            # Initialize for current segment
            d = 0
            current_point = t.waypoints[i]
            current_point_shapely = geo_primitive_to_shapely(current_point)

            while d <= segment_length:
                # Find minimum distance to all obstacles
                min_distance = float("inf")
                for obstacle in obstacles_shapely:
                    distance = obstacle.distance(current_point_shapely)
                    if distance <= r:  # Early collision
                        collision_idx.append(i)
                        break
                    min_distance = min(min_distance, distance)

                if i in collision_idx:
                    break

                # Calculate ssafety certificate
                safe_step = max(min_distance - r, r / 2)
                d += min(safe_step, segment_length - d)
                if segment_length - d < r / 2:
                    break

                current_point = get_interpolated_point(t.waypoints[i], t.waypoints[i + 1], d, segment_length)
                current_point_shapely = geo_primitive_to_shapely(current_point)

            # Final check
            if i not in collision_idx:
                end_point = geo_primitive_to_shapely(t.waypoints[i + 1])
                for obstacle in obstacles_shapely:
                    if obstacle.distance(end_point) <= r:
                        collision_idx.append(i)
                        break

        return collision_idx


# * MY STUFF


@dataclass
class GridCell:
    bounds: Tuple[float, float, float, float]
    occupied: bool = False


class OccupancyGrid:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, num_rows: int, num_cols: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size_x = (x_max - x_min) / num_cols
        self.cell_size_y = (y_max - y_min) / num_rows
        self.occupancy = np.zeros((num_rows, num_cols), dtype=bool)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        col = int((x - self.x_min) / self.cell_size_x)
        row = int((y - self.y_min) / self.cell_size_y)

        return (min(max(0, row), self.num_rows - 1), min(max(0, col), self.num_cols - 1))

    def update_occupancy(self, obstacles: List[shapely.geometry.Polygon]):
        self.occupancy.fill(False)
        for obstacle in obstacles:
            minx, miny, maxx, maxy = obstacle.bounds
            min_row, min_col = self.world_to_grid(minx, miny)
            max_row, max_col = self.world_to_grid(maxx, maxy)

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    world_x = self.x_min + col * self.cell_size_x
                    world_y = self.y_min + row * self.cell_size_y
                    # Check if point is inside obstacle
                    if obstacle.covers(shapely.Point(world_x, world_y)):
                        self.occupancy[row, col] = True

    def get_cells_for_polygon(self, poly: shapely.geometry.Polygon) -> npt.NDArray:
        """Get grid cells that a obstacle might intersect with"""
        minx, miny, maxx, maxy = poly.bounds
        min_row, min_col = self.world_to_grid(minx, miny)
        max_row, max_col = self.world_to_grid(maxx, maxy)
        intersecting_cells = []

        # Literally the same thing of update_occupancy()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                world_x = self.x_min + col * self.cell_size_x
                world_y = self.y_min + row * self.cell_size_y

                if poly.covers(shapely.Point(world_x, world_y)):
                    intersecting_cells.append([row, col])

        return np.array(intersecting_cells)

    def check_polygon_collision(self, poly: shapely.geometry.Polygon) -> bool:
        """Check if a polygon collides with any occupied cells"""
        cell_indices = self.get_cells_for_polygon(poly)
        # Quick check using array indexing
        if len(cell_indices) == 0:
            return False
        # Check if any cells in the region are occupied
        occupied = np.any(self.occupancy[cell_indices[:, 0], cell_indices[:, 1]])
        if occupied:
            return True
        return False
