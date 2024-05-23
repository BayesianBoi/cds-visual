import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from utils import create_output_dir, load_and_preprocess_data, plot_confusion_matrix, CIFAR10_LABELS

def perform_grid_search(X_train, y_train):
    """
    Grid search to find the best parameters
    """
    parameter_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "tol": [0.0001, 0.001, 0.01]
    }
    grid_search = GridSearchCV(LogisticRegression(solver="lbfgs", multi_class="multinomial"), parameter_grid, verbose=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def train_and_evaluate_model(X_train, y_train, X_test, y_test, C, tol):
    """
    Training, fitting and evaluating the model using the optimal parameters found in the gridsearch
    """
    clf = LogisticRegression(tol=tol, solver="lbfgs", C=C, multi_class="multinomial", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    report = classification_report(y_test, y_pred, target_names=CIFAR10_LABELS)
    accuracy = accuracy_score(y_test, y_pred)
    CMatrix = confusion_matrix(y_test, y_pred)
    
    return clf, accuracy, report, CMatrix

def main():
    create_output_dir()
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test, cifar10_labels = load_and_preprocess_data()
    
    # Uncomment to perform grid search to find the best hyperparameters
    # best_params = perform_grid_search(X_train, y_train)
    # print("Best parameters found: ", best_params)
    
    # Train and evaluate model
    clf, accuracy, report, CMatrix = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, 
        C=0.01, tol=0.0001
    )
    print(f"Model Accuracy: {accuracy:.2f}")
    print(report)
    
    # Save classification report
    with open("out/report_lr.txt", "w") as f:
        f.write(report)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(CMatrix, cifar10_labels, "Confusion Matrix - Logistic", "../out/CM_logistic.png")

if __name__ == "__main__":
    main()
