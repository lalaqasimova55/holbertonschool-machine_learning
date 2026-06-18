#!/usr/bin/env python3
"""
Decision Tree using Gini Impurity split criterion.
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

    def depth(self):
        """Returns maximum depth of the entire tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts total nodes or optionally just leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

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

    def np_extrema(self, arr):
        """Returns the minimum and maximum values of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Returns a feature and random threshold value to split upon."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    # ---------- GINI SPLIT METHODS ----------

    def possible_thresholds(self, node, feature):
        """Computes all viable threshold values for splitting a feature."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Calculates optimal threshold and lowest gini split for feature."""
        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return 0, np.inf

        sub_expl = self.explanatory[:, feature][node.sub_population]
        sub_target = self.target[node.sub_population]
        classes = np.unique(self.target)

        n = sub_expl.size

        # Vectorized array sizing configurations (n, t, c)
        feat_val = sub_expl[:, np.newaxis, np.newaxis]
        thresh_val = thresholds[np.newaxis, :, np.newaxis]
        targ_val = sub_target[:, np.newaxis, np.newaxis]
        class_val = classes[np.newaxis, np.newaxis, :]

        Left_F = (feat_val > thresh_val) & (targ_val == class_val)
        Right_F = (feat_val <= thresh_val) & (targ_val == class_val)

        card_left_c = np.sum(Left_F, axis=0)
        card_right_c = np.sum(Right_F, axis=0)
        card_left = np.sum(feat_val > thresh_val, axis=0)
        card_right = np.sum(feat_val <= thresh_val, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            gini_l = 1 - np.sum((card_left_c / card_left) ** 2, axis=1)
            gini_r = 1 - np.sum((card_right_c / card_right) ** 2, axis=1)

        gini_l[card_left.ravel() == 0] = 0
        gini_r[card_right.ravel() == 0] = 0

        gini_split = (card_left.ravel() / n) * gini_l + (
            card_right.ravel() / n) * gini_r

        best_idx = np.argmin(gini_split)
        return thresholds[best_idx], gini_split[best_idx]

    def Gini_split_criterion(self, node):
        """Evaluates best overall feature and threshold via Gini scores."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    # ----------------------------------------

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree on explanatory and target values."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        elif self.split_criterion == "Gini":
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """Recursively trains a single node or flags it as a leaf."""
        # Düzgün yarpaq (Leaf) qərarı və ssenari idarəetməsi
        if (node.depth == self.max_depth or
                np.unique(self.target[node.sub_population]).size == 1 or
                np.sum(node.sub_population) <= self.min_pop):
            return

        node.feature, node.threshold = self.split_criterion(node)
        if node.threshold == np.inf:
            return

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 == self.max_depth or
            np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates and configures a terminal leaf node child."""
        if np.sum(sub_population) == 0:
            value = np.argmax(np.bincount(self.target[node.sub_population]))
        else:
            value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates and configures a decision internal split child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Computes accurate ratio scores between models vs real targets."""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target
        )) / test_target.size
