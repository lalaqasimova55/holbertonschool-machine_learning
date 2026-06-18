#!/usr/bin/env python3
"""
Decision Tree - full implementation (Holberton final version)
"""

import numpy as np


class Node:
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
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
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

        for c in [self.left_child, self.right_child]:
            if c:
                c.update_bounds_below()

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        return

    def pred(self, x):
        return self.value


class Decision_Tree:
    def __init__(self, root=None, max_depth=10,
                 min_pop=1, seed=0,
                 split_criterion="random"):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    # ================= REQUIRED FIX =================
    def pred(self, x):
        return self.root.pred(x)

    def predict(self, X):
        return np.array([self.pred(x) for x in X])
    # ===============================================

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

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
        x = self.rng.random()
        threshold = (1 - x) * fmin + x * fmax
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(target, dtype=bool)

        self.fit_node(self.root)
        self.update_bounds()

        if verbose == 1:
            print("Training finished.\n")
            print(f"Depth : {self.depth()}")
            print(f"Number of nodes : {self.count_nodes()}")
            print(f"Number of leaves : {self.count_nodes(True)}")
            print(f"Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):

        if np.sum(node.sub_population) == 0:
            node.is_leaf = True
            return

        node.feature, node.threshold = self.split_criterion(node)

        left_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )

        right_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        # LEFT
        if (np.sum(left_pop) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[left_pop]).size == 1):

            value = np.argmax(np.bincount(self.target[left_pop]))
            node.left_child = Leaf(value, node.depth + 1)
        else:
            node.left_child = Node(depth=node.depth + 1)
            node.left_child.sub_population = left_pop
            self.fit_node(node.left_child)

        # RIGHT
        if (np.sum(right_pop) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[right_pop]).size == 1):

            value = np.argmax(np.bincount(self.target[right_pop]))
            node.right_child = Leaf(value, node.depth + 1)
        else:
            node.right_child = Node(depth=node.depth + 1)
            node.right_child.sub_population = right_pop
            self.fit_node(node.right_child)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
