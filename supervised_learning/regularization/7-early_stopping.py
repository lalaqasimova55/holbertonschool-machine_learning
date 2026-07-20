#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop gradient descent early.

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: minimum improvement threshold
        patience: maximum number of consecutive epochs
                  without sufficient improvement
        count: current count of epochs without sufficient improvement

    Returns:
        tuple:
            stop (bool): True if training should stop, False otherwise
            count (int): updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return (count >= patience, count)
