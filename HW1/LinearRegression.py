import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.001, epoch=1000, batch_size=16, n_features=1):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.n_features = n_features
        self.weights = None

    def init_weights(self):
        self.weights = np.random.randn(self.n_features + 1, 1)

    def MSE_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)/2

    def preprocess(self, X):
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        m, n = X.shape
        X_ = np.zeros((m, n+1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def gradient(self, X, y, y_pred):
        X_ = self.preprocess(X)
        return - X_.T @ (y.reshape(y_pred.shape)- y_pred)/X.shape[0]

    def predict(self, X):
        X_ = self.preprocess(X)
        return X_ @self.weights

    def iter_mini_batch(self, X, y):
        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        for i in range(0, X.shape[0], self.batch_size):
            yield X[indices[i:i+self.batch_size]], y[indices[i:i+self.batch_size]]

    def SGD(self, X, y):
        for n in range(self.epoch):
            index = np.random.choice(X.shape[0])
            X_choose = X[index]
            y_choose = y[index]
            y_pred = self.predict(X_choose)
            loss = self.MSE_loss(y_choose, y_pred)
            grad = self.gradient(X_choose, y_choose, y_pred)
            self.weights -= self.lr * grad
        return self.weights

    def BGD(self, X, y):
        for n in range(self.epoch):
            y_pred = self.predict(X)
            loss = self.MSE_loss(y, y_pred)
            grad = self.gradient(X, y, y_pred)
            self.weights -= self.lr * grad
        return self.weights
            
    def MBGD(self, X, y):
        for n in range(self.epoch):
            for X_batch, y_batch in self.iter_mini_batch(X, y):
                y_pred = self.predict(X_batch)
                loss = self.MSE_loss(y_batch, y_pred)
                grad = self.gradient(X_batch, y_batch, y_pred)
                self.weights -= self.lr * grad
        return self.weights
    
    def min_max_normalization(self, X):
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    def mean_normalization(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def mapping(self, X, method = None):
        if method == "min_max":
            w_norm = self.weights[1:] / (X.max(axis=0) - X.min(axis=0))
            b_norm = self.weights[0] - w_norm @ X.min(axis=0)
        elif method == "mean":
            w_norm = self.weights[1:] / X.std(axis=0)
            b_norm = self.weights[0] - w_norm @ X.mean(axis=0)
        return np.vstack([b_norm, w_norm])

    def OLS(self, X, y):
        X_ = self.preprocess(X)
        self.weights = np.linalg.inv(X_.T @ X_) @ X_.T @ y.reshape(-1, 1)
        return self.weights


