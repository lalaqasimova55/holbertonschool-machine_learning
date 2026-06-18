#!/usr/bin/env python3
"""
Decision Tree printing module.
"""

import numpy as np


class Node:
    """
    Decision tree internal node.
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
        self.depth = depth

    # ---------- PRINT HELPERS ----------

    def _add_prefix(self, text, prefix):
        lines = text.split("\n")
        res = ""
        for i, line in enumerate(lines):
            if i == 0:
                res += prefix + line + "\n"
            else:
                res += "| " + line + "\n"
        return res

    # ---------- STRING FORMAT ----------

    def __str__(self):
        name = "root" if self.is_root else "node"
        base = f"{name} [feature={self.feature}, threshold={self.threshold}]"

        left = ""
        right = ""

        if self.left_child:
            left = self.left_child.__str__()
            left = self._add_prefix(left, "+---> ")

        if self.right_child:
            right = self.right_child.__str__()
            right = self._add_prefix(right, "+---> ")

        if left and right:
            return base + "\n" + left + right
        if left:
            return base + "\n" + left
        if right:
            return base + "\n" + right

        return base

    # ---------- TREE LOGIC ----------

    def max_depth_below(self):
        if self.left_child is None and self.right_child is None:
            return self.depth

        left = self.left_child.max_depth_below() if self.left_child else self.depth
        right = self.right_child.max_depth_below() if self.right_child else self.depth

        return max(self.depth, left, right)

    def count_nodes_below(self, only_leaves=False):
        if self.left_child is None and self.right_child is None:
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
    Leaf node.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"leaf [value={self.value}]"

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1


class Decision_Tree:
    """
    Decision tree container.
    """

    def __init__(self, root=None, max_depth=10,
                 min_pop=1, seed=0,
                 split_criterion="random"):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def __str__(self):
        return str(self.root)
