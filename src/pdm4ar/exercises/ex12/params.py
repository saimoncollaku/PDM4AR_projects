from dataclasses import dataclass


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2
    dt: float = 0.1
    max_sample_time: float = 6.0
    min_sample_time: float = 2.0
    sample_delta_time: float = 0.5
    min_sample_speed: float = 5.0
    max_sample_speed: float = 25.0
    sdot_sample_space: float = 0.5  # sample distances of sdot values
    emergency_timesteps: int = 5
    start_planning_time: float = 0.2  # time when the planning actually starts, before that just observe
    lane_change_fraction: float = 1.0
    replan_del_t: float = 1.0
    max_min_ttc: float = 0.8
    max_track_error: float = 2.0
    eval_weights = [0.1, 5.0, 2.0, 0.005, 0.01]
    lower_limit_v = 2.5  # this is used for Dubin's and emergency
