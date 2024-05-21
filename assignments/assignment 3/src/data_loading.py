import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Contains different functions for loading and pre-processing the data
"""

def load_data(data_folder):
    """
    Loads image file paths and their corresponding labels from the given folder.
    """
    # get the list of category of documents in the folder
    categories = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

    # lists to store the labels and file paths of the images
    file_paths = []
    labels = []

    # looping through each category of files, and appending the individual files to that subfolder
    for category in categories: 
        category_path = os.path.join(data_folder, category)
        for file in os.listdir(category_path):
            if file.endswith(".jpg"): # there is another file type than jpg in the subfolders
                file_paths.append(os.path.join(category_path, file))
                labels.append(category)

    # convert the list to an array
    labels = np.array(labels)

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42) # splitting the data in 80/20 while making sure that the different classes are equally split (as the data set is quite imbalanced)

    return train_files, test_files, train_labels, test_labels, categories

def create_generators(train_files, test_files, train_labels, test_labels):
    """
    Creates training and test data generator
    """
    datagen = ImageDataGenerator(rescale=1./255) # standardiizing the data so that the range is [0, 1]

    train_generator = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": train_files, "class": train_labels}),
        x_col="filename",
        y_col="class",
        target_size=(224, 224), # reshaping to the required image size of imagenet
        batch_size=32, # applying batch size of 32 - been tinkering a lot with the batch size, and 32 was optimal
        class_mode="categorical",
        shuffle=True  # ensures that the data is shuffled so that the model sees the data in different order each epoch
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": test_files, "class": test_labels}),
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False # we want to keep the evaluation consistent, so no shuffling for the test set
    )

    return train_generator, test_generator
