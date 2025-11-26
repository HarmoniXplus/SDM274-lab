import numpy as np

class Metrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def precision_recall_f1(self, y_true, y_pred):
        precision_list = []
        recall_list = []
        f1_list = []
        for i in range(self.n_classes):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)  
            f1_list.append(f1_score)
        return precision_list, recall_list, f1_list
    
    def macro_f1(self, y_true, y_pred):
        _, _, f1_scores = self.precision_recall_f1(y_true, y_pred)
        return np.mean(f1_scores)
    
    def weighted_f1(self, y_true, y_pred):
        _, _, f1_scores = self.precision_recall_f1(y_true, y_pred)
        return np.sum(np.bincount(y_true) @ f1_scores / len(y_true))
    
    def confusion_matrix(self, y_true, y_pred):
        confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                confusion_matrix[i][j] = np.sum((y_true == i) & (y_pred == j))
        return confusion_matrix
    
    def evaluate(self, y_true, y_pred):
        accuracy = self.accuracy(y_true, y_pred)
        precision, recall, f1_scores = self.precision_recall_f1(y_true, y_pred)
        macro_f1 = self.macro_f1(y_true, y_pred)
        weighted_f1 = self.weighted_f1(y_true, y_pred)
        confusion_matrix = self.confusion_matrix(y_true, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_scores": f1_scores,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": confusion_matrix
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        import matplotlib.pyplot as plt

        cm = self.confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.n_classes)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
