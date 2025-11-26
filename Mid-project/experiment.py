import numpy as np
import time
from data_loader import MNISTLoader
from metrics import Metrics
from logistic_regression import MultiClassLogisticRegression
from mlp_classifier import MLP
from knn_classifier import KNNClassifier
from decision_tree import DecisionTree

def run_experiment():
    print("Loading and preprocessing data...")
    loader = MNISTLoader()
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.preprocess()
    loader.visualize_samples()
    print("Data ready!")

    metrics = Metrics(n_classes=10)
    results = {}

    print("\nTraining Logistic Regression...")
    logistic_model = MultiClassLogisticRegression(n_features=784, n_classes=10, learning_rate=0.01, reg_lambda=0.00001)
    start_time = time.time()
    logistic_loss, logistic_acc = logistic_model.fit(X_train, y_train, epoch=1000, batch_size=64)
    train_time = time.time() - start_time

    start_time = time.time()
    logistic_predict = logistic_model.predict(X_test)
    predict_time = time.time() - start_time

    logistic_metrics = metrics.evaluate(y_test, logistic_predict)
    results['Logistic Regression'] = {
        'metrics': logistic_metrics,
        'train_time': train_time,
        'predict_time': predict_time
    }
    logistic_model.plot_learning_curve(logistic_loss, logistic_acc)

    print("\nTraining MLP...")
    mlp_model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.05)
    start_time = time.time()
    mlp_loss, mlp_acc = mlp_model.fit(X_train, y_train, epoch=100, batch_size=64)
    train_time = time.time() - start_time

    start_time = time.time()
    mlp_predict = mlp_model.predict(X_test)
    predict_time = time.time() - start_time

    mlp_metrics = metrics.evaluate(y_test, mlp_predict)
    results['MLP'] = {
        'metrics': mlp_metrics,
        'train_time': train_time,
        'predict_time': predict_time
    }
    mlp_model.plot_learning_curve(mlp_loss, mlp_acc)

    print("\nTraining KNN Classifier...")
    knn_model = KNNClassifier(k=10)
    start_time = time.time()
    knn_model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    knn_predict = knn_model.predict(X_test)
    predict_time = time.time() - start_time

    knn_metrics = metrics.evaluate(y_test, knn_predict)
    results['KNN'] = {
        'metrics': knn_metrics,
        'train_time': train_time,
        'predict_time': predict_time
    }

    print("\nTraining Decision Tree...")
    dt_model = DecisionTree(max_depth=5)
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    dt_predict = dt_model.predict(X_test)
    predict_time = time.time() - start_time

    dt_metrics = metrics.evaluate(y_test, dt_predict)
    results['Decision Tree'] = {
        'metrics': dt_metrics,
        'train_time': train_time,
        'predict_time': predict_time
    }

    print("\nExperiment Results:")
    for model, result in results.items():
        print(f"{model}:")
        for key, value in result.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    run_experiment()