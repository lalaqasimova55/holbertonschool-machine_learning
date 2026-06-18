#!/usr/bin/env python3
"""
Random Forest classification model building block.
"""

import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    Random Forest ensemble classifier container.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the random forest parameters and storage."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Generates the majority voting class prediction for input data."""
        preds = []
        for pred_func in self.numpy_preds:
            preds.append(pred_func(explanatory))

        # Massivi (n_trees, n_samples) formasına gətiririk
        preds = np.array(preds)

        # Sətirlər nümunələri, sütunlar isə ağacların proqnozlarını göstərir
        preds_t = preds.T
        majority_votes = np.array([
            np.argmax(np.bincount(sample_preds))
            for sample_preds in preds_t
        ])

        return majority_votes

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Trains multiple random split decision trees to create the forest."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            # split_criterion="random" parametrini mütləq bura qeyd edirik
            T = Decision_Tree(
                split_criterion="random",
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}\n"
                  f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}\n"
                  f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}\n"
                  f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}\n"
                  f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """Computes total accurate predicted ratio scores."""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target
        )) / test_target.size
