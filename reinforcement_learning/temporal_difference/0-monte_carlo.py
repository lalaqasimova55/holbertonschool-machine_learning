#!/usr/bin/env python3
"""Monte Carlo policy evaluation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    env: environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes in a state and returns the next action to take
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Step 1: Simulate the episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Step 2: Backward pass to calculate G and update V
        G = 0
        visited_states = [step[0] for step in episode]
        for t, (state, reward) in enumerate(reversed(episode)):
            idx = len(episode) - 1 - t
            G = gamma * G + reward

            # First-visit MC check
            if state not in visited_states[:idx]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
