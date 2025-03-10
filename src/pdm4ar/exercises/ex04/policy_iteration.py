import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, Action, Cell
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid, dtype=float)
        policy = np.full_like(grid_mdp.grid, Action.ABANDON, dtype=Action)
        policy[grid_mdp.goal_state[0]] = Action.STAY

        while True:
            old_policy = policy.copy()
            value_func = policy_evaluation(policy, grid_mdp)
            policy = policy_improvement(value_func, grid_mdp)
            if np.all(old_policy == policy):
                break

        return value_func, policy


def policy_evaluation(policy: np.ndarray, grid_mdp: GridMdp) -> np.ndarray:
    V = np.zeros_like(grid_mdp.grid).astype(float)

    while True:
        old_V = V.copy()
        for state in np.ndindex(grid_mdp.grid.shape):
            if grid_mdp.grid[state] == Cell.CLIFF:
                continue
            action = Action(policy[state])
            V[state] = 0
            if action == Action.ABANDON:
                T = grid_mdp.get_transition_prob(state, action, grid_mdp.start_state[0])
                R = grid_mdp.stage_reward(state, action, grid_mdp.start_state[0])
                V[state] += (grid_mdp.gamma * old_V[grid_mdp.start_state[0]] + R) * T
            else:
                for next_state in grid_mdp.get_admissible_next_states(state):
                    T = grid_mdp.get_transition_prob(state, action, next_state)
                    R = grid_mdp.stage_reward(state, action, next_state)
                    V[state] += (grid_mdp.gamma * old_V[next_state] + R) * T

        if np.all(V - old_V < 4e-2):
            break
    return V


def policy_improvement(value_fun: np.ndarray, grid_mdp: GridMdp) -> np.ndarray:
    policy = np.zeros_like(grid_mdp.grid).astype(int)

    for state in np.ndindex(grid_mdp.grid.shape):
        possible_policies = {}
        if grid_mdp.grid[state] == Cell.CLIFF:
            continue
        for action in grid_mdp.get_admissible_actions(state):
            possible_policies[action] = 0

            if action == Action.ABANDON:
                T = grid_mdp.get_transition_prob(state, action, grid_mdp.start_state[0])
                R = grid_mdp.stage_reward(state, action, grid_mdp.start_state[0])
                possible_policies[action] += (grid_mdp.gamma * value_fun[grid_mdp.start_state[0]] + R) * T
            else:
                for next_state in grid_mdp.get_admissible_next_states(state):
                    T = grid_mdp.get_transition_prob(state, action, next_state)
                    R = grid_mdp.stage_reward(state, action, next_state)
                    possible_policies[action] += (grid_mdp.gamma * value_fun[next_state] + R) * T
        policy[state] = max(possible_policies, key=possible_policies.get)

    return policy
