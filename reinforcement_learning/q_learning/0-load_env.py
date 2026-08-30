#!/usr/bin/env python3
"""
Module to load the FrozenLake environment from Gymnasium.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Parameters:
    - desc: list of lists containing a custom description of the map
    - map_name: string containing the pre-made map to load
    - is_slippery: boolean determining if the ice is slippery

    Returns:
    - the gymnasium environment
    """
    if desc is None and map_name is None:
        env = gym.make(
            'FrozenLake-v1',
            map_name='8x8',
            is_slippery=is_slippery
        )
    elif desc is not None:
        env = gym.make(
            'FrozenLake-v1',
            desc=desc,
            is_slippery=is_slippery
        )
    else:
        env = gym.make(
            'FrozenLake-v1',
            map_name=map_name,
            is_slippery=is_slippery
        )

    return env
