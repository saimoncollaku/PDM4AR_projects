from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq  # you may find this helpful

import math
from typing import Dict, List, Tuple, Optional
from collections import deque

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass


@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # Init
        queue: List[Tuple[float, X]] = []
        heapq.heappush(queue, (0, start))
        parents: Dict[X, X | None] = {start: None}
        cost_to_reach: Dict[X, float] = {start: 0}
        opened: list[X] = []

        while queue:
            # Remove lowest item in queue
            current_cost, current = heapq.heappop(queue)
            opened.append(current)

            # Goal state
            if current == goal:
                path = reconstruct_path(parents, start, goal)
                return path

            # Other state
            adjacents = self.graph.adj_list.get(current, set())
            for adjacent in adjacents:
                new_cost = current_cost + self.graph.get_weight(current, adjacent)
                if adjacent not in parents:
                    parents[adjacent] = current
                    cost_to_reach[adjacent] = new_cost
                    heapq.heappush(queue, (new_cost, adjacent))
                elif new_cost < cost_to_reach[adjacent] and adjacent not in opened:
                    parents[adjacent] = current
                    cost_to_reach[adjacent] = new_cost
                    heapq.heappush(queue, (new_cost, adjacent))

        # Failed
        return []


@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, lon_goal: float, lat_goal: float) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, lon_goal, lat_goal)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, lon_goal: float, lat_goal: float) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        lon_u, lat_u = self.graph.get_node_coordinates(u)
        # lon_v, lat_v = self.graph.get_node_coordinates(v)

        dist = direct_distance(lat1=lat_u, lat2=lat_goal, lon1=lon_u, lon2=lon_goal)
        time = dist / TravelSpeed.SECONDARY

        return time

    def path(self, start: X, goal: X) -> Path:
        # Init
        queue: List[Tuple[float, X]] = []
        heapq.heappush(queue, (0, start))
        parents: Dict[X, X | None] = {start: None}
        cost_to_reach: Dict[X, float] = {start: 0}
        opened: list[X] = []
        lon_g, lat_g = self.graph.get_node_coordinates(goal)

        while queue:
            # Remove lowest item in queue
            _, current = heapq.heappop(queue)
            current_cost = cost_to_reach[current]
            opened.append(current)

            # Goal state
            if current == goal:
                path = reconstruct_path(parents, start, goal)
                return path

            # Other state
            adjacents = self.graph.adj_list.get(current, set())
            for adjacent in adjacents:
                new_cost = current_cost + self.graph.get_weight(current, adjacent)
                if adjacent not in parents:
                    parents[adjacent] = current
                    cost_to_reach[adjacent] = new_cost
                    heu = self.heuristic(adjacent, lat_goal=lat_g, lon_goal=lon_g)
                    heapq.heappush(queue, (new_cost + heu, adjacent))
                elif new_cost < cost_to_reach[adjacent] and adjacent not in opened:
                    parents[adjacent] = current
                    cost_to_reach[adjacent] = new_cost
                    heu = self.heuristic(adjacent, lat_goal=lat_g, lon_goal=lon_g)
                    heapq.heappush(queue, (new_cost + heu, adjacent))

        # Failed
        return []


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total


# * MY FUNS


def reconstruct_path(parents: Dict[X, Optional[X]], start: X, goal: X) -> Path:
    path = deque([])
    current: X = goal
    while current != start:
        path.appendleft(current)
        current = parents[current]
    path.appendleft(start)
    return list(path)


def direct_distance(lat1, lon1, lat2, lon2, earth_radius=6371000):
    """
    Calculate the direct distance THROUGH the Earth between two points.
    Optimized for speed.

    Parameters:
    lat1, lon1 : float
        Latitude and longitude of the first point in degrees
    lat2, lon2 : float
        Latitude and longitude of the second point in degrees
    earth_radius : float
        Radius of the Earth in meters (default is average radius: 6,371,000 meters)

    Returns:
    float: Direct distance THROUGH the Earth in meters
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    # Precompute trigonometric functions (no need for sine of longitude)
    sin_lat1, cos_lat1 = math.sin(lat1_rad), math.cos(lat1_rad)
    sin_lat2, cos_lat2 = math.sin(lat2_rad), math.cos(lat2_rad)
    # Calculate differences in trigonometric values
    cos_lon_diff = math.cos(lon2_rad - lon1_rad)
    sin_lon_diff = math.sin(lon2_rad - lon1_rad)
    # Use the precomputed trigonometric values to calculate dx, dy, dz
    dx = earth_radius * (cos_lat2 * cos_lon_diff - cos_lat1)
    dy = earth_radius * cos_lat2 * sin_lon_diff
    dz = earth_radius * (sin_lat2 - sin_lat1)
    # Return the Euclidean distance
    return math.sqrt(dx * dx + dy * dy + dz * dz)
