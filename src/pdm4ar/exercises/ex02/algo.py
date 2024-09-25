from abc import abstractmethod, ABC

from networkx import reconstruct_path
from sklearn import neighbors

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # ! DOING
        # Declaration
        queued = []
        visited = []
        parents = []
        
        # Init
        queued.append(start)
        visited.append(start)
        parents.append(None)
        
        while queued:
            current = queued.pop()
            if current == goal:
                # reconstruct path
                pass
            else:
                adjacents = list(AdjacencyList.get(current, set()))
                

                
            
        return [], []


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []
    
    
def get_adjacent_nodes(adjacency_list: Mapping[X, Set[X]], node: X) -> List[X]:
    adjacent_set = adjacency_list.get(node)
    return list(adjacent_set) if adjacent_set is not None else []