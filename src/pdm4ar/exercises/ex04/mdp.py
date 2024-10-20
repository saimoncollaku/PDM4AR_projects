from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc, Cell


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        self.gamma = gamma
        self.prob_wormhole = 1 / np.sum(self.grid == Cell.WORMHOLE)
        self.rows, self.cols = grid.shape

        self.transition_data: Dict[State, Dict[Action, Dict[State, float]]] = {}
        self.reward_data: Dict[State, Dict[Action, Dict[State, float]]] = {}
        self.admissible_actions_data: Dict[State, list[Action]] = {}
        self.admissible_next_states_data: Dict[State, list[State]] = {}
        self.start_state = list(zip(*np.where(self.grid == Cell.START)))
        self.goal_state = list(zip(*np.where(self.grid == Cell.GOAL)))
        self.wormhole_states = list(zip(*np.where(self.grid == Cell.WORMHOLE)))
        self.compute_all()

    def compute_all(self) -> None:
        for state in np.ndindex(self.grid.shape):
            # Skip cliffs
            if self.grid[state] == Cell.CLIFF:
                continue
            self.compute_admissable_actions(state)
            self.compute_admissible_next_states(state)

            for action in self.get_admissible_actions(state):
                if action == Action.ABANDON:
                    self.compute_reward(state, Action.ABANDON, self.start_state[0])
                    self.compute_transition_prob(state, Action.ABANDON, self.start_state[0])
                else:
                    for next_state in self.get_admissible_next_states(state):
                        self.compute_reward(state, action, next_state)
                        self.compute_transition_prob(state, action, next_state)

    def get_admissible_actions(self, state: State) -> list[Action]:
        return self.admissible_actions_data[state]

    def compute_admissable_actions(self, state: State) -> None:
        cell = self.grid[state]
        if cell == Cell.CLIFF:
            actions = [Action.ABANDON]
        elif cell == Cell.GOAL:
            actions = [Action.STAY]
        else:
            actions = self.get_admitted_directions(state)
            actions.append(Action.ABANDON)
        self.admissible_actions_data[state] = actions

    def get_admissible_next_states(self, state: State) -> list[State]:
        return self.admissible_next_states_data[state]

    def compute_admissible_next_states(self, state: State) -> None:
        cell = self.grid[state]
        next_states = []
        if cell == Cell.CLIFF:
            next_states = self.start_state
        elif cell == Cell.GOAL:
            next_states = [state]
        else:
            admitted_dir = self.get_admitted_directions(state)
            if self.check_adj_to_wormhole(state, admitted_dir[0], admitted_dir):
                next_states += self.wormhole_states
            elif self.check_dir_to_wormhole(state, admitted_dir[0], admitted_dir):
                next_states += self.wormhole_states
            if admitted_dir != 4:
                next_states += self.start_state
                if Action.NORTH in admitted_dir:
                    next_states.append((state[0] - 1, state[1]))
                if Action.SOUTH in admitted_dir:
                    next_states.append((state[0] + 1, state[1]))
                if Action.EAST in admitted_dir:
                    next_states.append((state[0], state[1] + 1))
                if Action.WEST in admitted_dir:
                    next_states.append((state[0], state[1] - 1))
            else:
                next_states.append((state[0] - 1, state[1]))
                next_states.append((state[0] + 1, state[1]))
                next_states.append((state[0], state[1] + 1))
                next_states.append((state[0], state[1] - 1))
            if cell == Cell.SWAMP:
                next_states.append(state)
                next_states += self.start_state
            next_states = list(set(next_states))
        self.admissible_next_states_data[state] = next_states

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        return self.transition_data[state][action][next_state]

    def compute_transition_prob(self, state: State, action: Action, next_state: State) -> None:
        self.transition_data.setdefault(state, {})
        self.transition_data[state].setdefault(action, {})
        prob = 0
        current_cell = self.grid[state]
        if current_cell != Cell.CLIFF:
            prob_fun = {
                Cell.GRASS: self.compute_prob_grass,
                Cell.WORMHOLE: self.compute_prob_grass,
                Cell.START: self.compute_prob_grass,
                Cell.SWAMP: self.compute_prob_swamp,
                Cell.GOAL: self.compute_prob_goal,
            }.get(current_cell, lambda *args: 0)
            prob = prob_fun(state, action, next_state)
        self.transition_data[state][action][next_state] = prob

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        return self.reward_data[state][action][next_state]

    def compute_reward(self, state: State, action: Action, next_state: State) -> None:
        self.reward_data.setdefault(state, {})
        self.reward_data[state].setdefault(action, {})
        reward = 0
        current_cell = self.grid[state]
        if current_cell != Cell.CLIFF:
            reward_func = {
                Cell.GRASS: self.compute_reward_grass,
                Cell.WORMHOLE: self.compute_reward_grass,
                Cell.START: self.compute_reward_grass,
                Cell.SWAMP: self.compute_reward_swamp,
                Cell.GOAL: self.compute_reward_goal,
            }.get(current_cell, lambda *args: 0)
            reward = reward_func(state, action, next_state)
        self.reward_data[state][action][next_state] = reward

    def compute_reward_grass(self, state: State, action: Action, next_state: State) -> float:
        next_cell = self.grid[next_state]
        admitted_dir = self.get_admitted_directions(state)

        if action == Action.ABANDON:
            return -10
        if next_cell == Cell.START:
            return -11 if len(admitted_dir) != 4 else -1
        if next_cell == Cell.WORMHOLE:
            if self.check_adj_to_wormhole(state, action, admitted_dir) != 0:
                return -1
            if self.check_dir_to_wormhole(state, action, admitted_dir):
                return -1
            return 0
        return -1

    def compute_reward_swamp(self, state: State, action: Action, next_state: State) -> float:
        next_cell = self.grid[next_state]
        admitted_dir = self.get_admitted_directions(state)

        if action == Action.ABANDON:
            return -10
        if next_cell == Cell.START:
            return -12 if len(admitted_dir) != 4 else -2
        if next_cell == Cell.WORMHOLE:
            if self.check_adj_to_wormhole(state, action, admitted_dir) != 0:
                return -2
            if self.check_dir_to_wormhole(state, action, admitted_dir):
                return -2
            return 0
        return -2

    def compute_reward_goal(self, state: State, action: Action, next_state: State) -> float:

        if action == Action.STAY and state == next_state:
            return 50
        return 0

    def compute_prob_grass(self, state: State, action: Action, next_state: State) -> float:
        next_cell = self.grid[next_state]
        admitted_dir = self.get_admitted_directions(state)

        if action == Action.ABANDON:
            return 1 if next_cell == Cell.START else 0

        if next_cell == Cell.START:
            prob_cliff = (4 - len(admitted_dir)) * (0.25 / 3)
            if self.check_next_state_is_dir(state, action, next_state, admitted_dir):
                return 0.75 + prob_cliff
            if self.check_next_state_is_adj(state, next_state, admitted_dir):
                return (0.25 / 3) + prob_cliff
            if action not in admitted_dir:
                return 0.75 + prob_cliff - (0.25 / 3)
            return prob_cliff

        if next_cell == Cell.WORMHOLE:
            p = 0
            if self.check_dir_to_wormhole(state, action, admitted_dir):
                p += 0.75 * self.prob_wormhole
            p += self.check_adj_to_wormhole(state, action, admitted_dir) * ((0.25 / 3)) * self.prob_wormhole
            return p

        if self.check_next_state_is_dir(state, action, next_state, admitted_dir):
            return 0.75

        return 0.25 / 3

    def compute_prob_swamp(self, state: State, action: Action, next_state: State) -> float:
        next_cell = self.grid[next_state]
        admitted_dir = self.get_admitted_directions(state)

        if action == Action.ABANDON:
            return 1 if next_cell == Cell.START else 0
        if action == Action.STAY:
            return 1 if self.check_same_state(state, next_state) else 0

        prob_cliff = (4 - len(admitted_dir)) * (0.25 / 3) + 0.05

        if next_cell == Cell.START:
            if self.check_next_state_is_dir(state, action, next_state, admitted_dir):
                return 0.5 + prob_cliff
            if self.check_next_state_is_adj(state, next_state, admitted_dir):
                return (0.25 / 3) + prob_cliff
            if action not in admitted_dir:
                return 0.5 + prob_cliff - (0.25 / 3)
            return prob_cliff

        if next_cell == Cell.WORMHOLE:
            p = 0
            if self.check_dir_to_wormhole(state, action, admitted_dir):
                p += 0.5 * self.prob_wormhole
            p += self.check_adj_to_wormhole(state, action, admitted_dir) * ((0.25 / 3)) * self.prob_wormhole
            return p

        if self.check_next_state_is_dir(state, action, next_state, admitted_dir):
            return 0.5
        if state == next_state:
            return 0.2

        return 0.25 / 3

    def compute_prob_goal(self, state: State, action: Action, next_state: State) -> float:

        if action == Action.STAY and state == next_state:
            return 1
        return 0

    def get_admitted_directions(self, state: State) -> list[Action]:
        directions = []
        for action, (dr, dc) in zip(
            [Action.SOUTH, Action.NORTH, Action.EAST, Action.WEST], [(1, 0), (-1, 0), (0, 1), (0, -1)]
        ):
            new_r, new_c = state[0] + dr, state[1] + dc
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols and self.grid[new_r, new_c] != Cell.CLIFF:
                directions.append(action)
        return directions

    def check_same_state(self, state: State, next_state: State) -> bool:
        return state == next_state

    def check_dir_to_wormhole(self, state: State, action: Action, admissible_dir: list[Action]) -> bool:
        if action not in admissible_dir:
            return False
        dr, dc = {Action.NORTH: (-1, 0), Action.SOUTH: (1, 0), Action.EAST: (0, 1), Action.WEST: (0, -1)}[action]
        return self.grid[state[0] + dr, state[1] + dc] == Cell.WORMHOLE

    def check_adj_to_wormhole(self, state: State, action: Action, admitted_dir: list[Action]) -> int:
        return sum(
            1
            for a in admitted_dir
            if a != action
            and self.grid[
                state[0] + {Action.NORTH: -1, Action.SOUTH: 1, Action.EAST: 0, Action.WEST: 0}[a],
                state[1] + {Action.NORTH: 0, Action.SOUTH: 0, Action.EAST: 1, Action.WEST: -1}[a],
            ]
            == Cell.WORMHOLE
        )

    def check_next_state_is_dir(
        self, state: State, action: Action, next_state: State, admitted_dir: list[Action]
    ) -> bool:
        if action not in admitted_dir:
            return False
        dr, dc = {Action.NORTH: (-1, 0), Action.SOUTH: (1, 0), Action.EAST: (0, 1), Action.WEST: (0, -1)}[action]
        return next_state == (state[0] + dr, state[1] + dc)

    def check_next_state_is_adj(self, state: State, next_state: State, admitted_dir: list[Action]) -> bool:
        return any(self.check_next_state_is_dir(state, action, next_state, admitted_dir) for action in admitted_dir)


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass
