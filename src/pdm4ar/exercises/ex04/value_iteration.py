import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Cell, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        while True:
            old_v = value_func.copy()

            for state in np.ndindex(grid_mdp.grid.shape):
                # Skip cliffs
                if grid_mdp.grid[state] == Cell.CLIFF:
                    continue

                Q = {}
                for action in grid_mdp.get_admissible_actions(state):
                    Q[action] = 0
                    if action == Action.ABANDON:
                        T = grid_mdp.get_transition_prob(state, action, grid_mdp.start_state[0])
                        R = grid_mdp.stage_reward(state, action, grid_mdp.start_state[0])
                        Q[action] += T * (R + grid_mdp.gamma * value_func[grid_mdp.start_state[0]])
                    else:
                        for next_state in grid_mdp.get_admissible_next_states(state):
                            T = grid_mdp.get_transition_prob(state, action, next_state)
                            R = grid_mdp.stage_reward(state, action, next_state)
                            Q[action] += T * (R + grid_mdp.gamma * value_func[next_state])

                value_func[state] = max(Q.values())
                policy[state] = max(Q, key=Q.get)

            if np.all(value_func - old_v < 1e-3):
                break

        return value_func, policy
