from abc import abstractmethod, ABC

from typing import Dict, Optional
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
            adjacents = sorted(graph.get(current, set()), reverse=True)
            for adjacent in adjacents:
                if adjacent not in parents:
                    queue.appendleft(adjacent)
                    parents[adjacent] = current

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
            adjacents = sorted(graph.get(current, set()))
            for adjacent in adjacents:
                if adjacent not in parents:
                    queue.append(adjacent)
                    parents[adjacent] = current

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
            max_d_reached = 1

            while queue:
                # Remove first item in queue
                current = queue.popleft()
                opened.append(current)

                # Goal state
                if current == goal:
                    path = reconstruct_path(parents, start, goal)
                    return path, opened

                # Other state
                if depths[current] < d:
                    adjacents = sorted(graph.get(current, set()), reverse=True)
                    for adjacent in adjacents:
                        if adjacent not in parents:
                            queue.appendleft(adjacent)
                            parents[adjacent] = current
                            depths[adjacent] = depths[current] + 1
                            max_d_reached = max(max_d_reached, depths[adjacent])

            if max_d_reached < d:
                return [], opened


def reconstruct_path(parents: Dict[X, Optional[X]], start: X, goal: X) -> Path:
    path = deque([])
    current: X = goal
    while current != start:
        path.appendleft(current)
        current = parents[current]
    path.appendleft(start)
    return list(path)
