#!/usr/bin/env python3
"""
Decision Tree module with correct printing format.
"""

import numpy as np


class Node:
    """
    Internal node of decision tree.
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
        Formats left subtree string.
        """
        lines = text.split("\n")
        new_text = ""
        for i, line in enumerate(lines):
            if i == 0:
                new_text += "+---> " + line + "\n"
            else:
                new_text += "|      " + line + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Formats right subtree string.
        """
        lines = text.split("\n")
        new_text = ""
        for i, line in enumerate(lines):
            if i == 0:
                new_text += "+---> " + line + "\n"
            else:
                new_text += "       " + line + "\n"
        return new_text

    def __str__(self):
        """
        String representation of node.
        """
        node_type = "root" if self.is_root else "node"
        base = f"{node_type} [feature={self.feature}, threshold={self.threshold}]"

        left = ""
        right = ""

        if self.left_child:
            left = self.left_child.__str__()
            left = self.left_child_add_prefix(left)

        if self.right_child:
            right = self.right_child.__str__()
            right = self.right_child_add_prefix(right)

        if left and right:
            return base + "\n" + left + right
        if left:
            return base + "\n" + left
        if right:
            return base + "\n" + right

        return base

    def max_depth_below(self):
        """
        Returns max depth.
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


class Leaf(Node):
    """
    Leaf node.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """
        Leaf string representation.
        """
        return f"leaf [value={self.value}]"

    def max_depth_below(self):
        """
        Leaf depth.
        """
        return self.depth


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
        return self.root.max_depth_below()

    def __str__(self):
        return self.root.__str__()
