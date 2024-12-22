from dataclasses import dataclass


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2
    dt: float = 0.1
    max_sample_time: float = 2.6
    min_sample_time: float = 2.0
    min_sample_speed: float = 5.0
    max_sample_speed: float = 25.0
    sdot_sample_space: float = 2.0  # sample distances of sdot values
    emergency_timesteps: int = 5
    start_planning_time: float = 0.2  # time when the planning actually starts, before that just observe
