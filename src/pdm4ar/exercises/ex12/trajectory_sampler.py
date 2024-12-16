# Trajectory samples
# Generate candidate trajectories - each trajectory has N (max number of samples), traj[N]  = goal

import numpy as np


class TrajectorySample:
    # there are 3 frames
    # Global: x, y
    # Frenet: d, s
    # Vehicle: vx, vy
    T: int  # number of samples
    dt: float
    x: np.ndarray
    xdot: np.ndarray
    xdotdot: np.ndarray
    xdotdotdot: np.ndarray
    y: np.ndarray
    ydot: np.ndarray
    ydotdot: np.ndarray
    ydotdotdot: np.ndarray
    d: np.ndarray
    ddot: np.ndarray
    ddotdot: np.ndarray
    ddotdotdot: np.ndarray
    s: np.ndarray
    sdot: np.ndarray
    sdotdot: np.ndarray
    sdotdotdot: np.ndarray
    kappa: np.ndarray
    kappadot: np.ndarray

    # between control signals, the simulator does ZeroOrderHold anyways
    # but these are not control signals, these will create control signals
    # from MPC
