import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import create_output_dir, plot_confusion_matrix, CIFAR10_LABELS

# Parameters inspired by this kaggle notebook: https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16

# Function to load and preprocess data for VGG16
def vgg16_load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train.flatten(), num_classes=10) # converting class vectors to binary class matrices 
    y_test = to_categorical(y_test.flatten(), num_classes=10) # converting class vectors to binary class matrices 
    
    # Preprocess images for VGG16
    X_train_preprocessed = preprocess_input(X_train) #preprocessing the input 
    X_test_preprocessed = preprocess_input(X_test) #preprocessing the input 
    
    return X_train_preprocessed, y_train, X_test_preprocessed, y_test

# Function to load the VGG16 model
def load_vgg16_model(input_shape): #using the same layers as the kaggle 
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu", name="hidden1"),
        Dropout(0.4), #extra dropout layer
        Dense(256, activation="relu", name="hidden2"),
        Dropout(0.4), #extra dropout layer
        Dense(10, activation="softmax", name="predictions")  # 10 classes for CIFAR-10
    ])
    base_model.trainable = False  # Freeze the convolutional base
    return model

def main():
    create_output_dir() #create the output folder if it doesnt already exiist
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = vgg16_load_and_preprocess_data()
    
    # Data augmentation using the same parameters as the kaggle 
    aug = ImageDataGenerator(
        rotation_range=20, 
        zoom_range=0.15, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.15,
        horizontal_flip=True, 
        fill_mode="nearest"
    )
    
    # Load and compile the model
    model = load_vgg16_model(input_shape=(32, 32, 3))
    sgd = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"]) #using categorical_crossentropy as the input is multi-class
    
    # early stopping
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit( #agaiin inspired by the kaggle notebook
        aug.flow(X_train, y_train, batch_size=128),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // 128,
        epochs=100,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report and confusion matrix
    report = classification_report(y_true_classes, y_pred_classes, target_names=CIFAR10_LABELS)
    accuracy = np.mean(y_true_classes == y_pred_classes)
    CMatrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(report)
    
    # Save classification report
    with open("../out/report_vgg16.txt", "w") as f:
        f.write(report)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(CMatrix, CIFAR10_LABELS, "Confusion Matrix - CNN", "../out/CM_vgg16.png")

if __name__ == "__main__":
    main()
