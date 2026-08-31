#!/usr/bin/env python3
"""Trains an agent that can play Atari's Breakout using keras-rl2"""
import numpy as np
import gymnasium as gym
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Permute, Activation
)
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class GymnasiumWrapper(gym.Wrapper):
    """
    Wraps a gymnasium environment so that it exposes the classic
    (pre-gymnasium) gym API expected by keras-rl2:
        reset() -> observation
        step(action) -> observation, reward, done, info
        render() -> renders the environment using the mode set at
            environment creation time
    """

    def reset(self, **kwargs):
        """Resets the environment and returns only the observation"""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """
        Steps the environment and combines the terminated/truncated
        flags into a single done flag
        """
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, *args, **kwargs):
        """Renders the environment"""
        return self.env.render()


class AtariProcessor(Processor):
    """Processes the observations and rewards for the Atari agent"""

    def process_observation(self, observation):
        """
        Resizes and converts an observation to grayscale to reduce
        the input size the network needs to handle
        """
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """Normalizes the batch of states to values between 0 and 1"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clips the reward between -1 and 1"""
        return np.clip(reward, -1., 1.)


def build_model(window_length, shape, actions):
    """
    Builds the convolutional neural network used to approximate the
    Q function
    window_length - the number of stacked frames given to the network
    shape - the (height, width) of a single (processed) frame
    actions - the number of actions the agent can take
    Returns: a keras Sequential model
    """
    input_shape = (window_length,) + shape

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))

    return model


def build_agent(model, actions):
    """
    Builds the DQN agent used to train the Q network
    model - the keras model to be used as the Q network
    actions - the number of actions the agent can take
    Returns: a compiled DQNAgent
    """
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
        value_test=.05, nb_steps=1000000
    )

    dqn = DQNAgent(
        model=model, nb_actions=actions, policy=policy, memory=memory,
        processor=processor, nb_steps_warmup=50000, gamma=.99,
        target_model_update=10000, train_interval=4, delta_clip=1.
    )
    dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

    return dqn


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5')
    env = GymnasiumWrapper(env)
    np.random.seed(123)
    env.reset(seed=123)
    actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, INPUT_SHAPE, actions)
    dqn = build_agent(model, actions)

    checkpoint_callback = ModelIntervalCheckpoint(
        'checkpoint.h5', interval=250000
    )

    dqn.fit(
        env, nb_steps=1750000, log_interval=10000, visualize=False,
        verbose=2, callbacks=[checkpoint_callback]
    )

    dqn.save_weights('policy.h5', overwrite=True)
