#!/usr/bin/env python3
"""Displays a game of Atari's Breakout played by a trained agent"""
import numpy as np
import gymnasium as gym

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

from train import (
    GymnasiumWrapper, AtariProcessor, build_model, INPUT_SHAPE,
    WINDOW_LENGTH
)


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = GymnasiumWrapper(env)
    np.random.seed(123)
    env.reset(seed=123)
    actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, INPUT_SHAPE, actions)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model, nb_actions=actions, policy=policy, memory=memory,
        processor=processor
    )
    dqn.compile('adam', metrics=['mae'])
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=5, visualize=True)
