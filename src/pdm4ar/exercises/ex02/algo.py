from abc import abstractmethod, ABC
from ast import List

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
        queued: list[X] = [start]
        visited: list[X] = [start]
        opened: list[X] = []
        parents = []
        parents.append(None)

        while queued:
            # Remove first item in queue
            current = queued.pop(0)
            opened.append(current)

            # Goal state
            if current == goal:
                path = construct(visited, parents)
                return path, opened

            # Other state
            to_queue: list[X] = []
            for adjacent in graph.get(current, set()):
                if adjacent not in visited:
                    visited.append(adjacent)
                    to_queue.append(adjacent)
                    parents.append(current)
            queued = to_queue + queued

        # Failed
        return [], opened


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


def construct(visited_nodes: list[X], parent_nodes: list[X]) -> list[X]:
    # Last of the visited nodes is the goal, start from there
    last_state = visited_nodes[-1]
    parent_state = parent_nodes[-1]
    path = [last_state]

    # Stop when the parent is empty (reached the start)
    while parent_state is not None:
        last_state = parent_state
        parent_state = parent_nodes[visited_nodes.index(last_state)]
        path.append(last_state)

    path.reverse()

    return path
