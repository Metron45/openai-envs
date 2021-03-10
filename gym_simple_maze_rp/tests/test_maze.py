import pytest

import gym
from gym.spaces import Discrete
from gym_simple_maze_rp.maze import SimpleMazeRP
from gym_yacs_simple_maze.maze import Action


class TestSimpleMazeRP:

    @pytest.fixture
    def env(self):
        return gym.make("SimpleMazeRP-v0")

    def test_should_perform_happy_path_and_get_paid_for_it(self, env):
        # given
        moves = [
            {'action': Action.NORTH, 'exp_state': '1001', 'exp_rew': 1},  # 3 -> 0
            {'action': Action.EAST, 'exp_state': '1010', 'exp_rew': 1},  # 0 -> 1
            {'action': Action.EAST, 'exp_state': '1100', 'exp_rew': 1},  # 1 -> 2
            {'action': Action.SOUTH, 'exp_state': '0101', 'exp_rew': 1},  # 2 -> 5
            {'action': Action.SOUTH, 'exp_state': '0110', 'exp_rew': 1},  # 5 -> 8
            {'action': Action.WEST, 'exp_state': '0010', 'exp_rew': 1},  # 8 -> 7
        ]

        # when
        env.reset()

        # then
        for step in moves:
            state, reward, done, _ = env.step(step['action'].value)
            print(f"{state} {reward} {done}")
            assert state == list(step['exp_state'])
            assert reward == step['exp_rew']
            assert done is False

        # final step
        state, reward, done, _ = env.step(Action.WEST.value)
        assert state == list('1011')
        assert reward == 100
        assert done is True
