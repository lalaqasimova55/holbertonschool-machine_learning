#!/usr/bin/env python3
"""Has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        total_rewards, rendered_outputs
            total_rewards: the rewards for the episode
            rendered_outputs: a list of rendered outputs representing
                the board state at each step
    """
    reset_res = env.reset()
    state = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        step_res = env.step(action)

        if len(step_res) == 5:
            new_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            new_state, reward, done, _ = step_res

        rendered_outputs.append(env.render())

        state = new_state
        total_rewards += reward

        if done:
            break

    return total_rewards, rendered_outputs
