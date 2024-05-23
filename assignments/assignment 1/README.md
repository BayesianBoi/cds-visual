# Assignment 1 - Building a simple image search algorithm

## Repository overview
This repository contains Python scripts to search for similar images using two different methods: colour histograms and a pretrained VGG16 model. Both methods extract and compare features from a chosen image to identify the top five most similar images to a chosen target image.

### Assignment objective
The objective of this task is to build a simple image search algorithm using a dataset of over 1000 images of flowers. The specific steps involved are:
1. Define a particular image that you want to work with
2. Extract the colour histogram of the target image using OpenCV
3. Extract colour histograms for all other images in the dataset
4. Compare the histogram of the target image to all other histograms using the `cv2.compareHist()` function with the `cv2.HISTCMP_CHISQR` metric
5. Find the five images which are most similar to the target image
6. Save a CSV file showing the five most similar images and their distance metrics
7. Repeat step 5-6 using a pre-trained CNN

## Data source
The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford. Download the dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and place the images in the `../in/flowers` directory

## Steps for running the analysis

### Setting up the environment
1. **Set up the virtual environment and install requirements:**
    ```bash
    bash setup.sh
    ```
2. **Activate the virtual environment:**
    ```bash
    source envVis1/bin/activate
    ```

### Colour histogram

- **Run the script:**
    ```bash
    python src/image-search-hist.py <chosen_image_number>
    ```
Takes an image (between 1 and 1360) as argument. Replace `<chosen_image_number>` with the number of the target image (e.g., `1` for `image_0001.jpg`)

### VGG16 classification

- **Run the script:**
    ```bash
    python src/image-search-VGG16.py <chosen_image_number>
    ```
Takes an image (between 1 and 1360) as argument. Replace `<chosen_image_number>` with the number of the target image (e.g., `1` for `image_0001.jpg`)

## Extra script

### Comparison between the results for the two methods
Comparison between the identified most similar pictures between the two models

- **Run the script:**
    ```bash
    python src/comparison.py
    ```
You need to run the scripts for both of the model before comparing the results

## Summary of results

### Colour histogram

#### Colour histogram results
The CSV file contains the Chi-Squared distances of the top five images predicted as most similar to `image 1`. The CSV-file can be found in `/out`

| Filename | Distance  |
|----------|-----------|
| image_0773.jpg | 190.14    |
| image_1316.jpg | 190.22    |
| image_0740.jpg | 190.63    |
| image_1078.jpg | 191.69    |
| image_0319.jpg | 191.88    |

Distance is the Chi-Squared distance from the target image

### VGG16

#### VGG16 results
The CSV file contains the Euclidean distances of the top five images predicted as most similar to `image 1`. The CSV-file can be found in `/out`

| Filename       | Distance  |
|----------------|-----------|
| image_0020.jpg | 55.67     |
| image_0040.jpg | 57.29     |
| image_0006.jpg | 57.57     |
| image_0013.jpg | 58.08     |
| image_0049.jpg | 58.15     |

Distance is the Euclidean distance from the target image

## Comparison
Euclidean and Chi-Squared distances are not comparable. So, a comparison was performed to find any overlaps between the top five most similar images predicted by both methods. The comparison shows no overlap for the predicted top five images between the two methods

### Colour histogram
![Colour Histogram Plot](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/hist_plot_0001.png)


### VGG16
![VGG16 Plot](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%201/out/vgg16_plot_0001.png)

## Limitations and possible improvements
### Limitations
#### Colour histogram
Colour histograms are insensitive to texture and shape; meaning it only compares colour distributions and does not consider texture, shape or other important visual features that contribute to image similarity. In this case, the chosen image (image #1) is of a daffodil. However, none of the top five most similar images are of a daffodil. This highlights the flaws of the colour histogram-comparison method and suggests that other methods might be more suitable for finding similar pictures.

### Possible improvements
#### Using more advanced methods
The VGG16 classification method outperformed the colour histogram method in identifying similar images. The VGG16 model correctly classified all five similar images while the colour histogram method did not correctly classify any of the predicted similar images.

