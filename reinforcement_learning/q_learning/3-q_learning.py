#!/usr/bin/env python3
"""
Module to perform Q-learning on a Gymnasium environment.
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning for a given environment and Q-table.

    Parameters:
    - env: the FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate
    - epsilon: initial threshold for epsilon greedy
    - min_epsilon: minimum value that epsilon should decay to
    - epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
    - Q: the updated Q-table
    - total_rewards: list containing the rewards per episode
    """
    initial_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            res = env.step(action)
            if len(res) == 5:
                next_state, reward, terminated, truncated, _ = res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = res
                terminated = done

            if terminated and reward == 0:
                reward = -1

            if done:
                Q[state, action] = Q[state, action] + alpha * (
                    reward - Q[state, action]
                )
            else:
                Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )

            state = next_state
            episode_reward += reward

            if done:
                break

        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

        total_rewards.append(episode_reward)

    return Q, total_rewards
