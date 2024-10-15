import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, State, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid, dtype=float)
        policy = np.full_like(grid_mdp.grid, Action.ABANDON, dtype=Action)
        policy[grid_mdp.start_state] = Action.STAY

        while True:
            old_policy = policy.copy()
            value_func = policy_evaluation(policy, grid_mdp)
            policy = policy_improvement(value_func, grid_mdp)
            if np.all(old_policy == policy):
                break

        return value_func, policy


def policy_evaluation(policy: np.ndarray, grid_mdp: GridMdp) -> np.ndarray:
    V = np.zeros_like(grid_mdp.grid).astype(float)
    epsilon = 1e-2

    while True:
        old_V = V.copy()
        for state in np.ndindex(grid_mdp.grid.shape):
            action = Action(policy[state])
            V[state] = 0
            for next_state in grid_mdp.get_admissible_next_states(state):
                T = grid_mdp.get_transition_prob(state, action, next_state)
                R = grid_mdp.stage_reward(state, action, next_state)
                V[state] += (grid_mdp.gamma * old_V[next_state] + R) * T

        if np.all(V - old_V < 1):
            break
    return V


def policy_improvement(value_fun: np.ndarray, grid_mdp: GridMdp) -> np.ndarray:
    policy = np.zeros_like(grid_mdp.grid).astype(int)

    for state in np.ndindex(grid_mdp.grid.shape):
        possible_policies = {}
        for action in grid_mdp.get_admissible_actions(state):
            possible_policies[action] = 0

            for next_state in grid_mdp.get_admissible_next_states(state):
                T = grid_mdp.get_transition_prob(state, action, next_state)
                R = grid_mdp.stage_reward(state, action, next_state)
                possible_policies[action] += (grid_mdp.gamma * value_fun[next_state] + R) * T
        policy[state] = max(possible_policies, key=possible_policies.get)

    return policy
