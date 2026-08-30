#!/usr/bin/env python3
"""
Module to play an episode in FrozenLake environment using trained Q-table.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using full exploitation.

    Parameters:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - max_steps: maximum number of steps in the episode

    Returns:
    - total_rewards: total rewards for the episode
    - rendered_outputs: list of rendered outputs representing board states
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    rendered_outputs = [env.render()]
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state])

        res = env.step(action)
        if len(res) == 5:
            next_state, reward, terminated, truncated, _ = res
            done = terminated or truncated
        else:
            next_state, reward, done, _ = res

        rendered_outputs.append(env.render())
        total_rewards += reward
        state = next_state

        if done:
            break

    return total_rewards, rendered_outputs
