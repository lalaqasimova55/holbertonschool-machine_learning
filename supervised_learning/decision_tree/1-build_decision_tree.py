#!/usr/bin/env python3
"""
Decision Tree module.

Implements Node, Leaf and Decision_Tree classes.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the maximum depth in the subtree rooted at this node.
        """
        if self.is_leaf or (
            self.left_child is None and self.right_child is None
        ):
            return self.depth

        left_depth = (
            self.left_child.max_depth_below()
            if self.left_child else self.depth
        )
        right_depth = (
            self.right_child.max_depth_below()
            if self.right_child else self.depth
        )

        return max(self.depth, left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts nodes in subtree.

        Args:
            only_leaves (bool): if True, counts only leaves.

        Returns:
            int: number of nodes.
        """
        if self.is_leaf:
            return 1

        if only_leaves:
            left = self.left_child.count_nodes_below(True) if self.left_child else 0
            right = self.right_child.count_nodes_below(True) if self.right_child else 0
            return left + right

        left = self.left_child.count_nodes_below(False) if self.left_child else 0
        right = self.right_child.count_nodes_below(False) if self.right_child else 0

        return 1 + left + right


class Leaf(Node):
    """
    Leaf node of a decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns depth of leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts leaf node as 1.

        Args:
            only_leaves (bool): unused for leaf.

        Returns:
            int: 1
        """
        return 1


class Decision_Tree:
    """
    Decision tree container class.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns tree depth.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts nodes in the tree.

        Args:
            only_leaves (bool): if True counts only leaves.

        Returns:
            int: number of nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
