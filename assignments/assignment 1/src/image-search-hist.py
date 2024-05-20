import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Functions
# Colour histogram calculator
def colour_hist(image):
    """
    Calculates the colour histogram of an input image.
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0,256, 0,256, 0,256]) # calculates the colour histogram of the image
    hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX) # min/max normalizes the histogram 
    hist = hist.flatten() # flattens the histogram
    return hist

# Histogram similarity checker
def hist_similarity(hist1, hist2):
    """
    Calculates the similarity between two colour histograms using chi-squared distance metric.
    """
    return round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR), 2)

# Resize image for niiice plotting
def resize_image(image, size=(300, 300)):
    """
   Resizes the input image to 300x300 to use in plotting the images afterwards.
    """
    return cv2.resize(image, size)

# Main pipeline- taking the chosen image number as input
def main(chosen_image_number):
    input_folder = os.path.join("..", "in", "flowers") # Path to the input folder containing the flower images
    output_folder = os.path.join("..", "out") # Path to the output folder

    # Loading the chosen image from the path
    chosen_image_filename = f"image_{int(chosen_image_number):04d}.jpg" # instead of taking "image_0001.jpg" as the input, it will simply take "1" as the input
    chosen_image_path = os.path.join(input_folder, chosen_image_filename)
    chosen_image = cv2.imread(chosen_image_path)
    print("Chosen image loaded successfully")

    # Check if the image is loaded properly
    if chosen_image is None:
        raise ValueError(f"Image not found. Choose another image between 1 and 1360")

    similar_images = [] # Creating a list to store the similarity

    # Calculate the colour histogram of the chosen picture
    chosen_hist = colour_hist(chosen_image)

    image_files = [image for image in os.listdir(input_folder) if image != chosen_image_filename] # Adding the images to a list excluding the chosen image
    total_images = len(image_files) #calculating the total number of images to use in the progress bar

    for idx, imagefile in enumerate(image_files, start=1): # Using index to add progress bar to the analysis
        image_path = os.path.join(input_folder, imagefile)
        image = cv2.imread(image_path)
        
        # check if the image is loaded properly
        if image is None:
            print(f"Image {imagefile} not found. Moving on to the next...")
            continue
        
        image_hist = colour_hist(image) # Calculating the color histogram for the picture
        similarity = hist_similarity(chosen_hist, image_hist) # Comparing the color histogram to the chosen picture
        similar_images.append((imagefile, similarity)) # Appending the picture filename and the similarity to the list

        # progress bar for every 100 images processed
        if idx % 100 == 0:
            print(f"Analysed {idx}/{total_images} of the images")

    similar_images.sort(key=lambda rank: rank[1]) # sorting by similarity which is second element

    top_five = similar_images[:5] # Ranking the top five most similar images

    # Save the results to CSV in the output folder
    csv_filename = f"hist_{chosen_image_filename}.csv"
    output_csv_path = os.path.join(output_folder, csv_filename)
    os.makedirs(output_folder, exist_ok=True) # makes output folder if it doesnt exist
    with open(output_csv_path, "w") as csv:
        csv.write("Filename,Distance\n") # creating columns with the filename and distance
        csv.write(f"CHOSEN PICTURE: {chosen_image_filename},0.0\n")
        for filename, distance in top_five: # adding the results to the csv
            csv.write(f"{filename},{distance}\n")


    # plot of the chosen image and the most similar images
    plt.figure(figsize=(18, 10))
    
    # plot the chosen image
    chosen_image_plot = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)
    chosen_image_plot = resize_image(chosen_image_plot)
    plt.subplot(1, 6, 1)
    plt.imshow(chosen_image_plot)
    plt.title(f"Chosen Image: {int(chosen_image_number)}")
    plt.axis("off")

    # Plotting the most similar images
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
    plot_filename = f"hist_plot_{chosen_image_filename}.png"
    output_plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(output_plot_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Search based on colour histograms")
    parser.add_argument("chosen_image_number", type=int, help="The number of the chosen image (e.g., 1 for image_0001.jpg)")
    args = parser.parse_args()
    main(args.chosen_image_number)
