import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        if activation == 'relu':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2./input_size)
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
            
        self.b = np.zeros(output_size)
        self.input = None
        self.output = None
        self.activation_input = None

    def forward(self, X):
        self.input = X
        self.activation_input = X @ self.W + self.b
        if self.activation == 'relu':
            self.output = np.maximum(0, self.activation_input)
        elif self.activation == 'softmax':
            exp_values = np.exp(self.activation_input - np.max(self.activation_input, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output
    
    def backward(self, grad_output):
        if self.activation == 'relu':
            grad_activation = grad_output * (self.activation_input > 0)
        elif self.activation == 'softmax':
            grad_activation = grad_output

        grad_input = grad_activation @ self.W.T / self.input.shape[0]
        grad_W = self.input.T @ grad_activation / self.input.shape[0]
        grad_b = np.sum(grad_activation, axis=0) / self.input.shape[0]

        return grad_input, grad_W, grad_b
    
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.layers = []
        self.layers.append(Layer(input_size, hidden_sizes[0], 'relu'))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(Layer(hidden_sizes[i], hidden_sizes[i+1], 'relu'))
        self.layers.append(Layer(hidden_sizes[-1], output_size, 'softmax'))
        self.learning_rate = learning_rate
        self.n_classes = output_size
        
    def to_one_hot(self, y):
        y = y.astype(int)
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def forward(self, X):
        output = X
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        grad_output = y_pred - y_true
        for i in range(len(self.layers)-1, -1, -1):
            grad_input , grad_W, grad_b = self.layers[i].backward(grad_output)
            #update
            self.layers[i].W -= self.learning_rate * grad_W
            self.layers[i].b -= self.learning_rate * grad_b
            grad_output = grad_input

    def entropy_loss(self, y_pred, y_true):
        esp = 1e-15
        y_pred = np.clip(y_pred, esp, 1-esp)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))   
     
    def predict(self, X):
        probability = self.forward(X)
        return np.argmax(probability, axis = 1)
    
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
            accuracy = correct / X.shape[0]
            epoch_loss.append(loss)
            epoch_acc.append(accuracy)

            if verbose and (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return epoch_loss, epoch_acc

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
    