#!/usr/bin/env python3
"""Monte Carlo algorithm module."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        # First-visit or every-visit Monte Carlo update (standard update per visited state)
        visited_states = set()
        for t, (s, r) in enumerate(episode_data):
            if s in visited_states:
                continue
            visited_states.add(s)

            # Calculate return G
            G = sum(gamma ** (i - t) * episode_data[i][1] for i in range(t, len(episode_data)))
            
            # Update value estimate
            V[s] = V[s] + alpha * (G - V[s])

    return V
