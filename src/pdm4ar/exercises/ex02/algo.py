from abc import abstractmethod, ABC

from typing import Dict, List, Optional
from collections import deque

# from networkx import reconstruct_path

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, [] if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # Init
        queue = deque([start])
        opened: list[X] = []
        parents: Dict[X, X | None] = {start: None}

        while queue:
            # Remove first item in queue
            current = queue.popleft()
            opened.append(current)

            # Goal state
            if current == goal:
                path = reconstruct_path(parents, start, goal)
                return path, opened

            # Other state
            to_queue = deque([])
            for adjacent in graph.get(current, set()):
                if adjacent not in parents:
                    to_queue.append(adjacent)
                    parents[adjacent] = current
            queue = deque(sorted(to_queue)) + queue

        # Failed
        return [], opened


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # Init
        queue = deque([start])
        opened: list[X] = []
        parents: Dict[X, X | None] = {start: None}

        while queue:
            # Remove first item in queue
            current = queue.popleft()
            opened.append(current)

            # Goal state
            if current == goal:
                path = reconstruct_path(parents, start, goal)
                return path, opened

            # Other state
            to_queue = deque([])
            for adjacent in graph.get(current, set()):
                if adjacent not in parents:
                    to_queue.append(adjacent)
                    parents[adjacent] = current
            queue = queue + deque(sorted(to_queue))

        # Failed
        return [], opened


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:

        d = 0
        while True:
            # * DFS with iteration on depth
            d += 1

            # Init
            queue = deque([start])
            opened: list[X] = []
            parents: Dict[X, X | None] = {start: None}
            depths: Dict[X, int] = {start: 1}

            while queue:
                # Remove first item in queue
                current = queue.popleft()
                opened.append(current)

                # Goal state
                if current == goal:
                    path = reconstruct_path(parents, start, goal)
                    return path, opened

                # Other state
                to_queue = deque([])
                if depths[current] < d:
                    for adjacent in graph.get(current, set()):
                        if adjacent not in parents:
                            to_queue.append(adjacent)
                            parents[adjacent] = current
                            depths[adjacent] = depths[current] + 1
                    queue = deque(sorted(to_queue)) + queue

            # Checks if the depth provided new states
            if d not in depths.values():
                # No path found
                return [], opened


def reconstruct_path(parents: Dict[X, Optional[X]], start: X, goal: X) -> Path:
    path: List[X] = []
    current: X = goal
    while current != start:
        path.append(current)
        # Just to solve the type hint None problem
        parent = parents[current]
        if parent is None:
            return []
        current = parent
    path.append(start)
    return list(reversed(path))
