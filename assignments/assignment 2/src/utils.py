import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
import cv2

# Defining the Cifar10 labels (they are numeric by default)
CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 

# making sure that the out folder exists
def create_output_dir(directory="out"):
    """
    Creates the output folder if it does not already exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_preprocess_data():
    """
    Loads the cifar10 data and preprocesses the data (flattening the labels, greyscale, normalize, reshape)
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # loading the cifar data
    y_train = y_train.flatten() #flatteniing the labels as they are 2 dimensional right now
    y_test = y_test.flatten() #flattening the labels as they are 2 dimensional right now
    
    # Converting to greyscale
    X_train_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    
    # Normalizing the images
    X_train_normalized = X_train_gray / 255.0
    X_test_normalized = X_test_gray / 255.0

    # Flattening the images
    X_train_processed = X_train_normalized.reshape(-1, 1024)
    X_test_processed = X_test_normalized.reshape(-1, 1024)
    
    return X_train_processed, y_train, X_test_processed, y_test, CIFAR10_LABELS

def plot_confusion_matrix(cm, labels, title, output_path):
    """
    Plots a confusion matrix
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels) # seaborn heatmap
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

