# Trajectory samples
# Generate candidate trajectories - each trajectory has N (maxi number of samples), traj[N]  = goal


from abc import ABC, abstractmethod
import numpy as np


class TrajectorySampler:
    """This heavily uses the already implemented sampler of the FRENETIX
    algorithm! https://github.com/TUM-AVS/Frenetix-Motion-Planner

    """

    def __init__(
        self,
        dt: float,
        maxi_sampling_number: int,
        t_mini: float,
        horizon: float,
        delta_d_mini: float,
        delta_d_maxi: float,
        d_ego_pos: bool,
    ):
        self.dt = dt
        self.maxi_sampling_number = maxi_sampling_number
        self.s_sampling_mode = False
        self.d_ego_pos = d_ego_pos

        self.t_mini = t_mini
        self.horizon = horizon
        self.t_sampling = None

        self.delta_d_mini = delta_d_mini
        self.delta_d_maxi = delta_d_maxi
        self.d_sampling = None

        self.v_sampling = None
        self.s_sampling = None


class Sampling(ABC):
    def __init__(self, mini: float, maxi: float, max_res: int):
        """Class inherited by time, lateral and longitudinal samplers

        Args:
            mini: mini sampling range.
            maxi: maxi sampling range.
            max_res: sampling range max_resolution/step
        """

        assert maxi >= mini
        assert isinstance(max_res, int)
        assert max_res > 0

        self.mini = mini
        self.maxi = maxi
        self.max_res = max_res
        self.sample_v = list()
        self._initialization()

    @abstractmethod
    def _initialization(self):
        pass

    def to_range(self, step: int = 0) -> set:
        """
        Obtain the sampling steps of a given sampling stage

        Args:
            step: The sampling step to receive (>=0)
            maxi: maxi sampling range.
            max_res: sampling range max_resolution/step

        Returns:
            The set of sampling steps for the queried step
        """
        assert 0 <= step < self.max_res

        return self.sample_v[step]


class TimeSampling(Sampling):
    def __init__(self, mini: float, maxi: float, max_res: int, dt: float):
        self.dt = dt
        super(TimeSampling, self).__init__(mini, maxi, max_res)

    def _initialization(self):
        """
        Generate sampling sets with progressively finer granularity

        Each stage reduces the step size, creating denser sampling
        """
        for k in range(self.max_res):
            step_size = int((1 / (k + 1)) / self.dt)
            sampling_points = np.arange(self.mini, self.maxi + self.dt, step_size * self.dt)
            sampling_points = np.round(sampling_points, 2)
            sampling_points = sampling_points[sampling_points <= round(self.maxi, 2)]
            self.sample_v.append(set(sampling_points))
