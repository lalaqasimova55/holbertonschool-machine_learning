#!/usr/bin/env python3
"""
Decision Tree printing, training and evaluation.
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

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves)

    def fit(self, explanatory, target, verbose=0):
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(target, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("Training finished.\n")

            print(f"Depth : {self.depth()}")
            print(f"Number of nodes : {self.count_nodes()}")
            print(f"Number of leaves : {self.count_nodes(only_leaves=True)}")
            print(f"Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    # --- PLACEHOLDERS (from previous tasks assumed exist) ---

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

        if np.sum(left_pop) == 0:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if np.sum(right_pop) == 0:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf = Leaf(value)
        leaf.depth = node.depth + 1
        return leaf

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.mean(self.predict(test_explanatory) == test_target)
