# Kinematic check
# Cost calculation
# Collision & trajectory check
# Score candidate trajectories - cost functions, kinematic check & collision check > Behaviour, Velocity, Occlusion Planner

import numpy as np
from scipy import constants
from scipy.integrate import simpson
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

from pdm4ar.exercises.ex12.trajectory_sampler import TrajectorySample


def scalar_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.stack([x, y], axis=1)
    pure_acceleration = np.linalg.norm(xy, ord=2, axis=1)
    return pure_acceleration


class KinematicsFilter:
    sp: VehicleParameters
    sg: VehicleGeometry
    kappadot_allowed: float
    kappa_allowed: float
    __trajectory: TrajectorySample

    def __init__(self) -> None:
        self.kappa_allowed = np.tan(self.sp.delta_max) / self.sg.wheelbase
        raise NotImplementedError()

    @property
    def trajectory(self):
        return self.__trajectory

    def check(self, trajectory: TrajectorySample):
        self.__trajectory = trajectory
        return self.acceleration_filter() and self.curvature_filter() and self.yaw_filter()

    def acceleration_filter(self):
        """
        a_max check
        there is no v_switch defined in the problem, so not considering

        feasiblity not checked in between the sample times
        """
        pure_acceleration = scalar_value(self.__trajectory.xdotdot, self.__trajectory.ydotdot)
        max_acceleration = np.max(pure_acceleration)
        min_acceleration = np.min(pure_acceleration)
        if self.sp.acc_limits[0] > min_acceleration or self.sp.acc_limits[1] < max_acceleration:
            return False
        return True

    def curvature_filter(self):
        max_kappa = np.max(np.abs(self.__trajectory.kappa))
        max_kappadot = np.max(np.abs(self.__trajectory.kappadot))
        if self.kappa_allowed < max_kappa or self.kappadot_allowed < max_kappadot:
            return False
        return True

    def yaw_filter(self):
        """
        Basically kappa within limits
        """
        return True


class Cost:
    sp: VehicleParameters
    sg: VehicleGeometry
    __trajectory: TrajectorySample
    v_ref: float

    def __init__(self) -> None:
        raise NotImplementedError()

    @property
    def trajectory(self):
        return self.__trajectory

    def penalize_acceleration(self):
        pure_acceleration = scalar_value(self.__trajectory.xdotdot, self.__trajectory.ydotdot)
        y = np.square(pure_acceleration)
        return simpson(y, dx=self.__trajectory.dt)

    def penalize_jerk(self):
        pure_jerk = scalar_value(self.__trajectory.xdotdotdot, self.__trajectory.ydotdotdot)
        y = np.square(pure_jerk)
        return simpson(y, dx=self.__trajectory.dt)

    def velocity_offset(self):
        pure_velocity = scalar_value(self.__trajectory.xdot, self.__trajectory.ydot)
        # penalize over second half of path
        from_idx = int(self.__trajectory.T / 2)
        cost = np.sum(np.abs(pure_velocity[from_idx:-1] - self.v_ref))
        cost += np.square(pure_velocity[-1] - self.v_ref)
        return constants

    def penalize_deviation_from_reference(self):
        pass

    def penalize_closeness_from_obstacles(self):
        pass
