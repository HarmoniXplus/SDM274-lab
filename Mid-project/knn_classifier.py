import numpy as np
from collections import namedtuple
import heapq

class KDNode:
    def __init__(self, point, label, axis=None, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, X, y):
        self.k = X.shape[1]  # 特征维度
        self.root = self._build_tree(X, y, 0)
    
    def _build_tree(self, X, y, depth):
        if len(X) == 0:
            return None
        
        # 选择划分轴
        axis = depth % self.k
        
        # 按照当前轴的值排序
        sorted_idx = np.argsort(X[:, axis])
        X = X[sorted_idx]
        y = y[sorted_idx]
        
        # 选择中位数作为分割点
        median_idx = len(X) // 2
        
        # 创建节点
        node = KDNode(
            point=X[median_idx],
            label=y[median_idx],
            axis=axis
        )
        
        # 递归构建左右子树
        node.left = self._build_tree(X[:median_idx], y[:median_idx], depth + 1)
        node.right = self._build_tree(X[median_idx + 1:], y[median_idx + 1:], depth + 1)
        
        return node
    
    def _distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def find_k_nearest(self, point, k):
        # 使用最大堆来维护k个最近邻
        max_heap = []
        
        def search(node, depth=0):
            if node is None:
                return
            
            distance = self._distance(point, node.point)
            
            # 如果堆中元素少于k个，直接加入
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-distance, node.label))
            # 否则，如果当前点比堆顶更近，更新堆
            elif -distance > max_heap[0][0]:
                heapq.heapreplace(max_heap, (-distance, node.label))
            
            axis = depth % self.k
            diff = point[axis] - node.point[axis]
            
            # 递归搜索更可能包含近邻的子树
            if diff <= 0:
                search(node.left, depth + 1)
                # 如果到超平面的距离小于当前最大距离，搜索另一子树
                if node.right and (len(max_heap) < k or abs(diff) < -max_heap[0][0]):
                    search(node.right, depth + 1)
            else:
                search(node.right, depth + 1)
                if node.left and (len(max_heap) < k or abs(diff) < -max_heap[0][0]):
                    search(node.left, depth + 1)
        
        search(self.root)
        # 返回标签列表
        return [label for _, label in sorted(max_heap, reverse=True)]

class KNNClassifier:
    def __init__(self, k=5, algorithm='kd_tree'):
        self.k = k
        self.algorithm = algorithm  # 'kd_tree' or 'brute'
        self.X_train = None
        self.y_train = None
        self.tree = None

    def get_params(self):
        return {'k': self.k, 'algorithm': self.algorithm}
    
    def set_params(self, **params):
        if 'k' in params:
            self.k = params['k']
        if 'algorithm' in params:
            self.algorithm = params['algorithm']
        return self

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        if self.algorithm == 'kd_tree':
            self.tree = KDTree(X, y)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def find_neighbors_brute(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        return self.y_train[k_indices]
    
    def find_neighbors(self, x):
        if self.algorithm == 'kd_tree':
            return self.tree.find_k_nearest(x, self.k)
        else:  # 'brute'
            return self.find_neighbors_brute(x)
    
    def predict_sample(self, x=None, neighbors=None):
        if neighbors is None:
            neighbors = self.find_neighbors(x)
        vote = np.bincount(neighbors).argmax()
        return vote
    
    def predict(self, X, neighbors=None):
        if neighbors is None:
            return np.array([self.predict_sample(x) for x in X])
        else:
            return np.array([self.predict_sample(neighbors=neighbor) for neighbor in neighbors])