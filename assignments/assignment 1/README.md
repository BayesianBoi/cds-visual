# Image Search Using Colour Histograms

## Description
This repository contains a Python script to find similar images based on colour histograms. The script extracts and compares histograms using chi-squared distance to identify the top five most similar images to a chosen target image.

## Objective
The objective of this task is to build a simple image search algorithm using a dataset of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford. The specific steps involved are:
1. Choose a target image to work with.
2. Extract the colour histogram of the target image using OpenCV.
3. Extract colour histograms for all other images in the dataset.
4. Compare the histogram of the target image to all other histograms using the `cv2.compareHist()` function with the `cv2.HISTCMP_CHISQR` metric.
5. Find the five images most similar to the target image.
6. Save a CSV file showing the five most similar images and their distance metrics.
7. Save a plot showing the target image and the five most similar images.

## Data Source
The dataset consists of 1360 images of flowers from 17 species, provided by the Visual Geometry Group at the University of Oxford. Download the dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and place the images in the `../in/flowers` directory.

## Rerunning the Analysis
1. Run the setup in bash
```
sh setup.sh
```
2. Run the script
```
python image_search.py <chosen_image_number>
```
Replace <chosen_image_number> with the number of the target image (e.g., 1 for image_0001.jpg)


## Outputs (found in */out*)
### CSV File
The CSV file contains filenames and chi-squared distances of the top five similar images.

#### Results
| Image ID | Distance |
|------------|----------|
| 754        | 327.47   |
| 1079       | 354.01   |
| 532        | 360.85   |
| 771        | 361.43   |
| 747        | 362.21   |

Top five most similar to image 120, which was chosen as the target image.
Distance is the Chi-Squared distance from the target image.

### Plot
The plot shows the chosen target image and the top five similar images.

## Limitations
Insensitive to Texture and Shape: The method only compares colour distributions and does not consider texture, shape, or other important visual features that contribute to image similarity. For instance, the chosen image (image #120) is of a snowdrop. However, none of the top five most similar images are of a snowdrop. This highlights the flaws of the histogram-comparison method, and suggests that other methods might be more suitable for finding similar pictures

## Possible steps
One way to increase the accuracy of the model could be to use pretrained CNNs such as VGG16 to extract high-level features from the images. Otherwise, even a regressional approach using logistic or NN regression would probably increase the accuracy of the image search.
