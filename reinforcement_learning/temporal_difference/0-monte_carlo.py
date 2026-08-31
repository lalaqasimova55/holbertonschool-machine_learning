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
            episode_data.append((state, reward, terminated or truncated))

            if terminated or truncated:
                break
            state = next_state

        # First-visit Monte Carlo update
        visited_states = set()
        G = 0
        # Iterate backwards to compute returns easily
        for t in range(len(episode_data) - 1, -1, -1):
            s, r, done = episode_data[t]
            # In standard first-visit MC for episodic tasks:
            # If the episode ends with a hole or success, reward is given at the terminal step.
            # Let's compute return correctly.
            G = gamma * G + r
            
            if s not in visited_states:
                visited_states.add(s)
                V[s] = V[s] + alpha * (G - V[s])

    return V
