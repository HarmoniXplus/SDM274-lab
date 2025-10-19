import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.001, epoch=1000, batch_size=16):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weights = None

    def init_weights(self, n_features):
        self.weights = np.random.randn(n_features + 1, 1)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

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
        return X_.T @ (y_pred - y.reshape(y_pred.shape))/X.shape[0]

    def predict(self, X):
        X_ = self.preprocess(X)
        return self.sigmoid(X_ @ self.weights)

    def iter_mini_batch(self, X, y):
        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        for i in range(0, X.shape[0], self.batch_size):
            yield X[indices[i:i+self.batch_size]], y[indices[i:i+self.batch_size]]

    def fit(self, X, y, update_method='mini_batch'):
        self.init_weights(X.shape[1])
        
        for i in range(self.epoch):
            if update_method == 'stochastic':
                self.batch_size = 1
            for X_batch, y_batch in self.iter_mini_batch(X, y): 
                y_pred = self.predict(X_batch)
                gradient = self.gradient(X_batch, y_batch, y_pred)
                self.weights -= self.lr * gradient
        return self.weights

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    data = data[data[:, 0] != 3]
    X = data[:, 1:]
    y = data[:, 0]
    y = np.where(y == 1, 0, 1)
    return X, y.reshape(-1, 1)

def split_data(X, y, test_size=0.3):
    indice = list(range(X.shape[0]))
    np.random.shuffle(indice)
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indice[:split_idx]], X[indice[split_idx:]]
    y_train, y_test = y[indice[:split_idx]], y[indice[split_idx:]]
    return X_train, X_test, y_train, y_test

def evaluate(y_true, y_pred):
    y_pred = (y_pred >= 0.5).astype(int)
    acurracy = np.sum(y_true == y_pred) / y_true.shape[0]
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    F1_score = 2 * (precision * recall) / (precision + recall)
    return acurracy, precision, recall, F1_score

if __name__ == '__main__':
    X, y = load_data('./HW2/wine.data')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    # Mini-batch training
    model_MBGD = LogisticRegression(learning_rate=0.0001, epoch=5000, batch_size=16)
    model_MBGD.fit(X_train, y_train, update_method='mini_batch')
    y_MBGD_pred = model_MBGD.predict(X_test)

    accuracy, precision, recall, F1_score = evaluate(y_test, y_MBGD_pred)
    print("Mini-batch Gradient Descent Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {F1_score:.4f}")

    # Stochastic training
    model_SGD = LogisticRegression(learning_rate=0.0001, epoch=5000)
    model_SGD.fit(X_train, y_train, update_method='stochastic')
    y_SGD_pred = model_SGD.predict(X_test)
    
    accuracy, precision, recall, F1_score = evaluate(y_test, y_SGD_pred)
    print("\nStochastic Gradient Descent Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {F1_score:.4f}")
