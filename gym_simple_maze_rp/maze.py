from collections import namedtuple
from copy import copy
from enum import unique, IntEnum

import gym
from gym.spaces import Discrete
from gym_yacs_simple_maze.maze import Action, SimpleMaze


ActionState = namedtuple('ActionState', "action state")


class SimpleMazeRP(SimpleMaze):

    FINISH_LINE = 6

    # reward ideally should be natural
    # f.e. closer to end the better
    REWARDS = {
        0: [0, 1, 0, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: [1, 0, 0, 0],
        4: [0, 0, 1, 0],
        5: [0, 0, 1, 0],
        6: [],
        7: [0, 0, 0, 100],
        8: [0, 0, 0, 1]
    }

    def step(self, action):
        assert action in list(map(int, Action))

        reward = self.REWARDS[self._position][action]

        for transition in self.TRANSITIONS[self._position]:
            if transition.action == action:
                self._position = transition.state

        if self._position == self.FINISH_LINE:
            return self._current_perception(), reward, True, None

        return self._current_perception(), reward, False, None
