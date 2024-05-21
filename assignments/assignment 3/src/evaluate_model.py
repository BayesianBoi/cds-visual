import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

"""
Contains different functions for evaluating the model and plotting the results
"""

def evaluate_model(model, test_generator, history):
    """
    Evaluates the model on the test set and plots the results
    """
    # evaluate the model on the test set 
    test_loss, test_acc = model.evaluate(test_generator)

    # predictions for the test set
    predictions = model.predict(test_generator)

    # predicted label for the test set
    y_pred = np.argmax(predictions, axis=1)

    # true class for the test set
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()) # making the report for the accuracy
    print(report)

    with open("out/report.txt", "w") as f:
        f.write(report)

    # plotting training & vali loss values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    # Plot training and valid accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.savefig("out/loss_curve.png")
    plt.show()

    # making a confusion matrix for the predicted/true label
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", 
                xticklabels=test_generator.class_indices.keys(), 
                yticklabels=test_generator.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set Accuracy")
    plt.savefig("out/confusion_matrix.png")
