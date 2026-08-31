#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                 gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_history = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_history.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        G = 0
        for t in reversed(range(len(episode_history))):
            state, reward = episode_history[t]
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])
    return V
