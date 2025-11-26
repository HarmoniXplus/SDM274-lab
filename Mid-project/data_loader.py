import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

class MNISTLoader:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        print("Loading MNIST dataset...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        self.X = X
        self.y = y.astype(np.int32)
        print("Dataset loaded successfully!")

    def preprocess(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # normalization
        X_normalized = (self.X - np.mean(self.X, axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0) + 1e-8)

        # data split
        np.random.seed(42)
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(self.X.shape[0] * 0.8)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        self.X_train, self.X_test = X_normalized[train_idx], X_normalized[test_idx]
        self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
        print("Data preprocessing completed!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_samples(self):
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            idx = np.where(self.y == i)[0]
            sample_idx = np.random.choice(idx)
            ax = axes[i // 5, i % 5]
            ax.imshow(self.X[sample_idx].reshape(28, 28), cmap='gray')
            ax.set_title(f"Digit: {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()


    
