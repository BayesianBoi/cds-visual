import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from utils import create_output_dir, load_and_preprocess_data, plot_confusion_matrix, CIFAR10_LABELS

# Functions
# Grid search. However, we have already found the most optimal parameters, so not using it anymore
def NN_grid_search(X_train, y_train):
    """
    For finding the most optimal parameters
    """
    parameter_grid = {
        "hidden_layer_sizes": [(50,), (100,), (100, 100), (200,)],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "alpha": [0.0001, 0.001, 0.01]
    }
    grid_search = GridSearchCV(MLPClassifier(solver="adam", activation="relu", max_iter=200, early_stopping=True), parameter_grid, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def train_and_evaluate_nn(X_train, y_train, X_test, y_test, cifar10_labels, hidden_layer_sizes, learning_rate_init, alpha, max_iter = 1000, verbose=True, early_stopping=True):
    """
    Training, fitting and evaluating the model using the optimal parameters found in the gridsearch
    """
    nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver="adam", activation="relu", learning_rate_init=learning_rate_init, alpha=alpha, max_iter=max_iter, verbose=verbose, early_stopping=early_stopping)
    nn_model.fit(X_train, y_train)
    
    y_pred = nn_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=cifar10_labels)
    accuracy = accuracy_score(y_test, y_pred)
    CMatrix = confusion_matrix(y_test, y_pred)
    
    return nn_model, accuracy, report, CMatrix

def plot_loss_curve(nn_model, output_path="../out/loss_curve_nn.png"):
    """
    For plotting the loss curve
    """
    plt.plot(nn_model.loss_curve_)
    plt.xlabel("# Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(output_path)
    plt.show()

def main():
    create_output_dir() #create the "out" folder if does not already exiist
    
    X_train, y_train, X_test, y_test, cifar10_labels = load_and_preprocess_data()
    
    # We have already found the most optimal hyperparameters to be "alpha": 0.01, "hidden_layer_sizes": (200,), "learning_rate_init": 0.001
    # best_params = NN_grid_search(X_train, y_train)
    # print(best_params)
    
    nn_model, accuracy, report, CMatrix = train_and_evaluate_nn(
        X_train, y_train, X_test, y_test, cifar10_labels,
        hidden_layer_sizes=(200,),
        learning_rate_init=0.001,
        alpha= 0.01
    )
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(report)
    
    with open("../out/report_nn.txt", "w") as f:
        f.write(report)
    
    plot_loss_curve(nn_model)
    plot_confusion_matrix(CMatrix, cifar10_labels, "Confusion Matrix - NN", "../out/CM_nn.png")

if __name__ == "__main__":
    main()
