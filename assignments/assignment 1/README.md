# Image Search Using Colour Histograms and VGG16

## Repository Overview
This repository contains Python scripts to find similar images using two different methods: colour histograms and a pretrained VGG16 model. Both methods extract and compare features to identify the top five most similar images to a chosen target image.

### Assignment Objective
The objective of this task was to build a simple image search algorithm using a dataset of over 1000 images of flowers. The specific steps involved are:
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

### Colour Histogram Method
1. Run the setup in bash:
    ```bash
    sh setup.sh
    ```
2. Run the script:
    ```bash
    python image_search.py <chosen_image_number>
    ```
   Replace `<chosen_image_number>` with the number of the target image (e.g., `1` for `image_0001.jpg`).

### VGG16 Classification Method
1. Run the setup in bash:
    ```bash
    sh setup.sh
    ```
2. Run the script:
    ```bash
    python cnn_classification.py <chosen_image_number>
    ```
   Replace `<chosen_image_number>` with the number of the target image (e.g., `1` for `image_0001.jpg`).

## Outputs (found in */out*)

### Colour Histogram Method

#### CSV File
The CSV file contains filenames and chi-squared distances of the top five similar images.

#### Results
| Filename | Distance  |
|----------|-----------|
| image_0773.jpg | 190.14    |
| image_1316.jpg | 190.22    |
| image_0740.jpg | 190.63    |
| image_1078.jpg | 191.69    |
| image_0319.jpg | 191.88    |

Top five most similar to image 1 (chosen as the target image). Distance is the Chi-Squared distance from the target image.

#### Plot
The plot shows the chosen target image and the top five similar images predicted by comparing colour histograms


![Colour Histogram Results](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/hist_plot_0001.png)

### VGG16 Classification Method

#### CSV File
The CSV file contains filenames and Euclidean distances of the top five similar images.

#### Results
| Filename       | Distance  |
|----------------|-----------|
| image_0020.jpg | 55.67     |
| image_0040.jpg | 57.29     |
| image_0006.jpg | 57.57     |
| image_0013.jpg | 58.08     |
| image_0049.jpg | 58.15     |

Top five most similar to image 1 (chosen as the target image). Distance is the Euclidean distance from the target image.

#### Plot
The plot shows the chosen target image and the top five similar images predicted by VGG16.

![VGG16 Results](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/vgg16_plot_0001.png)

## Rank Comparison
A rank comparison was performed to evaluate the overlap between the top similar images identified by both methods. The comparison shows no overlap between the chosen pictures using the two methods.

### Colour Histogram
![Colour Histogram Plot](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/hist_plot_0001.png)


### VGG16 Classification 
![VGG16 Plot](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/vgg16_plot_0001.png)

## Limations and Possible Improvements

### Limitations
#### Colour Histogram Method
The colour historigram method is insensitive to texture and shape- meaning it only compares colour distributions and does not consider texture, shape, or other important visual features that contribute to image similarity. For instance, the chosen image (image #1) is of a daffodil. However, none of the top five most similar images are of a daffodil. This highlights the flaws of the colour histogram-comparison method and suggests that other methods might be more suitable for finding similar pictures.

### Possible Improvements
#### Using more advanced methods
The VGG16 classification method outperformed the colour histogram method in identifying similar images. The VGG16 model correctly classified all five similar images, while the colour histogram method did not correctly classify any of the similar images.

