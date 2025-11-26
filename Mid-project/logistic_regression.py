import numpy as np

class MultiClassLogisticRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01, reg_lambda=0.001):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.X = None

    def to_one_hot(self, y):
        y = y.astype(int)
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def forward(self, X):
        self.X = X
        z = X @ self.W + self.b
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return output
    
    def backward(self, y_true, y_pred):
        grad = y_pred - y_true
        grad_W = self.X.T @ grad / self.X.shape[0] + self.reg_lambda * self.W
        grad_b = np.sum(grad, axis=0) / self.X.shape[0]
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b

    def entropy_loss(self, y_pred, y_true):
        esp = 1e-15
        y_pred = np.clip(y_pred, esp, 1-esp)
        origin_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))
        l2_loss = 0.5 * self.reg_lambda * np.sum(self.W**2)
        return origin_loss + l2_loss

    def fit(self, X, y, epoch=1000, batch_size=32, verbose=True):
        epoch_loss = []
        epoch_acc = []
        y = self.to_one_hot(y)
        indices = np.arange(X.shape[0])
        for i in range(epoch):
            np.random.shuffle(indices)
            loss, correct = 0, 0
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[indices[j:j+batch_size]]
                y_batch = y[indices[j:j+batch_size]]
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                loss += self.entropy_loss(y_pred, y_batch)
                correct += np.sum(y_pred.argmax(axis=1) == y_batch.argmax(axis=1))
            loss = loss / (X.shape[0] // batch_size)
            epoch_loss.append(loss)
            accuracy = correct / X.shape[0]
            epoch_acc.append(accuracy)

            if verbose and (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return epoch_loss, epoch_acc

    def predict(self, X):
        probability = self.forward(X)
        return np.argmax(probability, axis = 1)

    def plot_learning_curve(self, epoch_loss, epoch_acc, title='Learning Curve'):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(range(1, len(epoch_loss) + 1), epoch_loss, color='tab:red')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')

        axes[1].plot(range(1, len(epoch_acc) + 1), epoch_acc, color='tab:blue')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()