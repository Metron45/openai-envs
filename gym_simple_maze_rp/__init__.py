from gym.envs.registration import register
from gym_simple_maze_rp.maze import SimpleMaze # noqa: F401

register(
    id='SimpleMazeRP-v0',
    entry_point='gym_simple_maze_rp.maze:SimpleMazeRP',
    max_episode_steps=1000,
    nondeterministic=False
)
