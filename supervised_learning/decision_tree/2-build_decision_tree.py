#!/usr/bin/env python3
"""
Decision Tree module with string visualization.
"""

import numpy as np


class Node:
    """
    Internal node of a decision tree.
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

    def left_child_add_prefix(self, text):
        """
        Adds prefix for left child visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds prefix for right child visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """
        String representation of a node.
        """
        left_str = ""
        right_str = ""

        if self.left_child:
            left_str = self.left_child.__str__()
            left_str = self.left_child_add_prefix(left_str)

        if self.right_child:
            right_str = self.right_child.__str__()
            right_str = self.right_child_add_prefix(right_str)

        node_str = f"node [feature={self.feature}, threshold={self.threshold}]"

        if left_str and right_str:
            return node_str + "\n" + left_str + right_str
        if left_str:
            return node_str + "\n" + left_str
        if right_str:
            return node_str + "\n" + right_str

        return node_str

    def max_depth_below(self):
        """
        Returns max depth of subtree.
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
        """
        if self.is_leaf:
            return 1

        if only_leaves:
            left = (
                self.left_child.count_nodes_below(True)
                if self.left_child else 0
            )
            right = (
                self.right_child.count_nodes_below(True)
                if self.right_child else 0
            )
            return left + right

        left = (
            self.left_child.count_nodes_below(False)
            if self.left_child else 0
        )
        right = (
            self.right_child.count_nodes_below(False)
            if self.right_child else 0
        )

        return 1 + left + right


class Leaf(Node):
    """
    Leaf node of decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        String representation of leaf.
        """
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """
        Returns leaf depth.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Leaf counts as 1.
        """
        return 1


class Decision_Tree:
    """
    Decision tree container.
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
        Counts nodes in tree.
        """
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        """
        String representation of tree.
        """
        return self.root.__str__()
