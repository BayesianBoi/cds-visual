import os
import numpy as np
import pandas as pd
import cv2
import argparse
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import matplotlib.pyplot as plt

# Functions
def load_model():
    """
    Loads the VGG16 model
    """
    base_model = VGG16(weights="imagenet") # using imagenet pretrained model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)
    return model

def preprocess_image(img_path):
    """
    Preprocesses the input image for VGG16 by resizing the image to the VGG16 model's required 224x224, converting it to array and
    expanding its dimensions.
    """
    img = image.load_img(img_path, target_size=(224, 224)) # Resizing to 224x224 pixels
    img_data = image.img_to_array(img) #converting to array
    img_data = np.expand_dims(img_data, axis=0) #expanding dimensions
    img_data = preprocess_input(img_data) # preprocessing
    return img_data

def extract_features(model, img_data):
    """
    Extracts the features from the images
    """
    features = model.predict(img_data)
    return features

def resize_image(image, size=(300, 300)):
    """
    Resizes the input image to 300x300 to use in plotting the images afterwards.
    """
    return cv2.resize(image, size)


# Main pipeline
def main(chosen_image_number):
    input_folder = os.path.join("in", "flowers") # Path to the input folder containing flower images
    output_folder = os.path.join("out") # Path to the output folder

    # Loading the chosen image
    chosen_image_filename = f"image_{int(chosen_image_number):04d}.jpg" # instead of taking "image_0001.jpg" as the input, it will simply take "1" as the input
    chosen_image_path = os.path.join(input_folder, chosen_image_filename)
    chosen_image = preprocess_image(chosen_image_path) #preprocess the chosen image
    print("Chosen image loaded successfully")

    # check if the image is loaded properly
    if chosen_image is None:
        raise ValueError(f"Image not found. Choose another number between 1 and 1360")

    # Load vgg16
    model = load_model()

    # extract the features for the chosen image
    chosen_image_features = extract_features(model, chosen_image)

    # making a list to store the results
    results = []

    # Process each image in the input folder
    for img_file in os.listdir(input_folder):
        if img_file != chosen_image_filename: #exclude the chosen image from the processing
            img_path = os.path.join(input_folder, img_file)
            img_data = preprocess_image(img_path)
            img_features = extract_features(model, img_data)
            
            distance = np.linalg.norm(chosen_image_features - img_features) # calculate similarity in euclidean distance
            results.append((img_file, distance))

    # Sort results by distance
    results.sort(key=lambda x: x[1])

    # get the top 5 most similar images
    top_five = results[:5]

    # Save the results to CSV in the output folder
    csv_filename = f"vgg16_{chosen_image_filename}.csv"
    output_csv_path = os.path.join(output_folder, csv_filename)
    os.makedirs(output_folder, exist_ok=True) # makes output folder if it doesnt exist
    with open(output_csv_path, "w") as csv:
        csv.write("Filename,Distance\n") # Creating columns with the filename and distance
        csv.write(f"CHOSEN PICTURE: {chosen_image_filename},0.0\n")
        for filename, distance in top_five: # Adding the results to the CSV
            csv.write(f"{filename},{distance}\n")

    # make plot of the chosen image and the most similar images
    plt.figure(figsize=(18, 10))
    
    # plotting the chosen image
    chosen_image_plot = cv2.imread(chosen_image_path)
    chosen_image_plot = cv2.cvtColor(chosen_image_plot, cv2.COLOR_BGR2RGB)
    chosen_image_plot = resize_image(chosen_image_plot)
    plt.subplot(1, 6, 1)
    plt.imshow(chosen_image_plot)
    plt.title(f"Chosen Image: {int(chosen_image_number)}")
    plt.axis("off")

    # plotting the most similar images
    for i, (filename, _) in enumerate(top_five, start=2):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image_to_plot = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_to_plot = resize_image(image_to_plot)
        plt.subplot(1, 6, i)
        plt.imshow(image_to_plot)
        image_number = filename
        plt.title(f"{image_number}")
        plt.axis("off")

    # Save the plot to the output folder
    plot_filename = f"vgg16_plot_{chosen_image_filename}.png"
    output_plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(output_plot_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification using VGG16")
    parser.add_argument("chosen_image_number", type=int, help="The number of the chosen image (e.g., 1 for image_0001.jpg)")

    args = parser.parse_args()
    main(args.chosen_image_number)
