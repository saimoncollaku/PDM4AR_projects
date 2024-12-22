import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from dg_commons.sim.goals import PlanningGoal, RefLaneGoal
from dg_commons.sim import SimObservations, InitSimObservations
from commonroad.scenario.lanelet import LaneletNetwork


def obtain_complete_ref(goal: RefLaneGoal, lanelet_network: LaneletNetwork) -> tuple[np.ndarray, int]:
    """
    Obtain the full reference trajectory by connecting the input reference points' lanelet
    with its predecessors and successors in a continuous order. Returns the complete path
    and the ID of the first lanelet.

    :param init_obs: Initial simulation observations containing the goal reference.
    :return: A tuple (complete_path, first_lanelet_id):
             - complete_path: Numpy array of shape (N, 2) with the complete trajectory points.
             - first_lanelet_id: The ID of the first lanelet (with no predecessor).
    """
    # Step 1: Extract reference points and find corresponding lanelet IDs
    ptx = goal.ref_lane.control_points
    x = [c.q.p[0] for c in ptx]
    y = [c.q.p[1] for c in ptx]

    # Create reference points for querying
    reference_points = np.column_stack((np.linspace(x[0], x[-1], 10), np.linspace(y[0], y[-1], 10)))
    point_list = [np.array(point) for point in reference_points]

    lane_ids = lanelet_network.find_lanelet_by_position(point_list=point_list)
    lane_ids = {lid[0] for lid in lane_ids if lid}  # Remove duplicates

    if not lane_ids:
        raise ValueError("No lanelet IDs found for the given reference points.")

    # Step 2: Find the starting lanelet (traverse backwards to find no-predecessor lanelet)
    lanelet_network = lanelet_network
    current_id = next(iter(lane_ids))  # Start with one lanelet ID
    first_lanelet_id = current_id

    while True:
        lanelet = lanelet_network.find_lanelet_by_id(current_id)
        if not lanelet.predecessor:
            first_lanelet_id = current_id  # Store the first lanelet ID
            break  # Found the starting lanelet (no predecessor)
        current_id = lanelet.predecessor[0]  # Move to predecessor

    # Step 3: Traverse forward through successors and build the complete path
    complete_path = []
    current_id = first_lanelet_id  # Start building the path from the first lanelet
    while current_id:
        lanelet = lanelet_network.find_lanelet_by_id(current_id)
        if not complete_path:
            complete_path.extend(lanelet.center_vertices)
        else:
            # Avoid duplicating the connection point
            if np.allclose(complete_path[-1], lanelet.center_vertices[0]):
                complete_path.extend(lanelet.center_vertices[1:])
            else:
                complete_path.extend(lanelet.center_vertices)

        # Move to the next successor
        if lanelet.successor:
            current_id = lanelet.successor[0]  # Move to the first successor
        else:
            current_id = None  # No more successors

    # Step 4: Ensure at least 5 waypoints in the path
    complete_path = np.array(complete_path)
    if len(complete_path) < 5:
        start_point = complete_path[0]
        end_point = complete_path[-1]
        # Generate 5 evenly spaced points including start and end
        complete_path = np.linspace(start_point, end_point, 5)

    return complete_path, first_lanelet_id


def get_lanelet_distances(lanelet_network: LaneletNetwork, target_id: int) -> tuple[float, float, float]:
    """
    Calculate distances between the target lanelet and its leftmost, rightmost, and a specified lanelet.

    :param lanelet_network: The LaneletNetwork containing all lanelets.
    :param target_id: The ID of the target lanelet.
    :param other_lane_id: The ID of another lanelet to compute its distance from the target lanelet.
    :return: A dictionary with distances to the leftmost, rightmost, and specified lanelet.
    """

    def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute the Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)

    def find_leftmost_lanelet(start_lanelet: Lanelet) -> Lanelet:
        """Find the leftmost lanelet by iterating adj_left until None."""
        current_lanelet = start_lanelet
        while current_lanelet.adj_left:
            if not current_lanelet.adj_left_same_direction:
                break
            current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
        return current_lanelet

    def find_rightmost_lanelet(start_lanelet: Lanelet) -> Lanelet:
        """Find the rightmost lanelet by iterating adj_right until None."""
        current_lanelet = start_lanelet
        while current_lanelet.adj_right:
            if not current_lanelet.adj_right_same_direction:
                break
            current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet.adj_right)
        return current_lanelet

    # Retrieve the target lanelet
    target_lanelet = lanelet_network.find_lanelet_by_id(target_id)
    if not target_lanelet:
        raise ValueError(f"Target lanelet ID {target_id} not found.")

    target_start_point = target_lanelet.center_vertices[0]  # First point of the target lanelet

    # Find leftmost lanelet and compute distance
    leftmost_lanelet = find_leftmost_lanelet(target_lanelet)
    leftmost_start_point = leftmost_lanelet.center_vertices[0]
    distance_to_leftmost = compute_distance(target_start_point, leftmost_start_point)

    # If there is no adj_left, the leftmost distance should be 0
    if not target_lanelet.adj_left:
        distance_to_leftmost = 0.0

    # Find rightmost lanelet and compute distance
    rightmost_lanelet = find_rightmost_lanelet(target_lanelet)
    rightmost_start_point = rightmost_lanelet.center_vertices[0]
    distance_to_rightmost = compute_distance(target_start_point, rightmost_start_point)

    # If there is no adj_right, the rightmost distance should be 0
    if not target_lanelet.adj_right:
        distance_to_rightmost = 0.0

    # Retrieve the specified other lanelet and compute distance
    if target_lanelet.adj_left != None:
        other_lanelet = lanelet_network.find_lanelet_by_id(target_lanelet.adj_left)
    else:
        other_lanelet = lanelet_network.find_lanelet_by_id(target_lanelet.adj_right)
    other_start_point = other_lanelet.center_vertices[0]

    # fix
    center1 = target_lanelet.center_vertices
    center2 = other_lanelet.center_vertices
    distances = np.linalg.norm(center1[:, np.newaxis, :] - center2[np.newaxis, :, :], axis=2)
    distance_to_other_lanelet = compute_distance(target_start_point, other_start_point)
    distance_to_other_lanelet = np.min(distances)

    return distance_to_leftmost, distance_to_rightmost, distance_to_other_lanelet
