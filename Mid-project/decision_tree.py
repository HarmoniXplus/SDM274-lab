import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 特征的索引
        self.threshold = threshold  # 分割阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的类别

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # 如果所有样本属于同一类别，或达到最大深度，创建叶节点
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return Node(value=self._most_common_label(y))

        # 寻找最佳分割
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=self._most_common_label(y))

        # 创建左右子树
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        # 遍历所有特征和阈值，寻找最佳分割
        best_feature = None
        best_threshold = None
        best_gain = 0
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        # 计算信息增益
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        child_entropy = (np.sum(left_indices) * left_entropy + np.sum(right_indices) * right_entropy) / len(y)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        num = np.bincount(y)
        probs = num / num.sum()
        return -np.sum(p * np.log2(p) for p in probs if p > 0)

    def _most_common_label(self, y):
        # 返回最常见的类别
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)