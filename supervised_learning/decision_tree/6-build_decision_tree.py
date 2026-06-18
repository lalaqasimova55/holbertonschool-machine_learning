#!/usr/bin/env python3
"""
Decision Tree printing, leaf retrieval, bounds, indicators, and predictions.
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

    # ---------- TREE LOGIC ----------

    def max_depth_below(self):
        """Returns max depth below node."""
        if self.left_child is None and self.right_child is None:
            return self.depth

        left = (self.left_child.max_depth_below()
                if self.left_child else self.depth)
        right = (self.right_child.max_depth_below()
                 if self.right_child else self.depth)

        return max(self.depth, left, right)

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes below node."""
        if self.left_child is None and self.right_child is None:
            return 1

        if only_leaves:
            left = (self.left_child.count_nodes_below(True)
                    if self.left_child else 0)
            right = (self.right_child.count_nodes_below(True)
                     if self.right_child else 0)
            return left + right

        left = (self.left_child.count_nodes_below(False)
                if self.left_child else 0)
        right = (self.right_child.count_nodes_below(False)
                 if self.right_child else 0)

        return 1 + left + right

    def get_leaves_below(self):
        """Returns the list of all leaves under this node."""
        leaves = []
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Recursively computes lower and upper feature bounds for nodes."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        if self.left_child is not None:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child is not None:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Computes and assigns the indicator function for this node."""
        def is_large_enough(x):
            """Checks if values are strictly greater than lower bounds."""
            comparisons = [np.greater(x[:, k], self.lower[k])
                           for k in self.lower.keys()]
            return np.all(np.array(comparisons), axis=0)

        def is_small_enough(x):
            """Checks if values are less than or equal to upper bounds."""
            comparisons = [np.less_equal(x[:, k], self.upper[k])
                           for k in self.upper.keys()]
            return np.all(np.array(comparisons), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0
        )

    def pred(self, x):
        """Traverses tree iteratively for a single profile evaluation."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """Returns max depth below leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes below leaf."""
        return 1

    def get_leaves_below(self):
        """Returns a list containing the leaf itself."""
        return [self]

    def update_bounds_below(self):
        """Leaf implementation of update_bounds_below."""
        pass

    def pred(self, x):
        """Returns target leaf valuation."""
        return self.value


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

    def get_leaves(self):
        """Returns a list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates lower and upper bounds starting from the root node."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Generates a fully vectorized dataset prediction lambda function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.value * leaf.indicator(A) for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        """Evaluates prediction on a single user input array profile."""
        return self.root.pred(x)
