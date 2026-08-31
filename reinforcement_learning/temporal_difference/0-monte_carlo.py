```python
#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns an action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()[0]
        states = []
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        G = 0

        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]

            if states[t] not in states[:t]:
                V[states[t]] += alpha * (G - V[states[t]])

    return V
```

