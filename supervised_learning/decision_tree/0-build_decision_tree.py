class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtree rooted at this node.

        The method recursively explores the left and right children (if they exist)
        and returns the greatest depth found among all descendant nodes, including
        the current node itself.

        Returns:
            int: The maximum depth value in the subtree.
        """

        if self.is_leaf or (self.left_child is None and self.right_child is None):
            return self.depth

        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth

        return max(self.depth, left_depth, right_depth)
