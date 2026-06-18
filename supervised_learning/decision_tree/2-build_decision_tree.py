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

    def left_child_add_prefix(self, text):
        """Adds standard prefix lines for a left child node."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                pfx = "    |      " if x.startswith(" ") else "    |  "
                new_text += (pfx + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds trailing indentation prefix lines for a right child node."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                pfx = "           " if x.startswith(" ") else "       "
                new_text += (pfx + x) + "\n"
        return new_text

    # ---------- STRING FORMAT ----------

    def __str__(self):
        """Returns the string representation of a node and its children."""
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child is not None:
            left_str = str(self.left_child)
            if self.is_root:
                out += ("    +---> " +
                        left_str.replace("\n", "\n    |  ") + "\n")
            else:
                lines = left_str.split("\n")
                out += "\n    +---> " + lines[0]
                for x in lines[1:]:
                    if x:
                        out += "\n    |  " + x

        if self.right_child is not None:
            right_str = str(self.right_child)
            if self.is_root:
                out += ("    +---> " +
                        right_str.replace("\n", "\n       ") + "\n")
            else:
                lines = right_str.split("\n")
                out += "\n    +---> " + lines[0]
                for x in lines[1:]:
                    if x:
                        out += "\n       " + x

        return out.strip("\n")


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
        """Returns string representation of a Leaf."""
        return f"leaf [value={self.value}]"

    def max_depth_below(self):
        """Returns max depth below leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes below leaf."""
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
        """Returns string representation of the tree from the root."""
        return self.root.__str__() + "\n"
