#!/usr/bin/env python3
"""
Decision Tree printing, leaf retrieval, bounds, indicators, and training.
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

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "    |      " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += "           " + x + "\n"
        return new_text

    def __str__(self):
        name = "root" if self.is_root else "node"
        base = f"{name} [feature={self.feature}, threshold={self.threshold}]"

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

    def get_leaves_below(self):
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        def is_large_enough(x):
            comps = [np.greater(x[:, k], self.lower[k])
                     for k in self.lower.keys()]
            return np.all(np.array(comps), axis=0)

        def is_small_enough(x):
            comps = [np.less_equal(x[:, k], self.upper[k])
                     for k in self.upper.keys()]
            return np.all(np.array(comps), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0
        )

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
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
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def pred(self, x):
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
        return self.root.__str__() + "\n"

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.value * leaf.indicator(A) for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        return self.root.pred(x)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            fmin, fmax = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = fmax - fmin
        x = self.rng.uniform()
        threshold = (1 - x) * fmin + x * fmax
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(target, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"Training finished.\n"
                f"Depth : {self.depth()}\n"
                f"Number of nodes : {self.count_nodes()}\n"
                f"Number of leaves : {self.count_nodes(only_leaves=True)}\n"
                f"Accuracy on training data : "
                f"{self.accuracy(self.explanatory, self.target)}",
                end=""
            )

    def fit_node(self, node):
        node.feature, node.threshold = self.split_criterion(node)

        left_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        left_leaf = (
            np.sum(left_pop) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[left_pop]).size == 1
        )

        if left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        right_leaf = (
            np.sum(right_pop) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[right_pop]).size == 1
        )

        if right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf = Leaf(value)
        leaf.depth = node.depth + 1
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.mean(self.predict(test_explanatory) == test_target)
