# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## Repository Overview
This repository contains scripts to classify the CIFAR-10 dataset using Logistic Regression and Neural Networks. Additionally, a CNN model using VGG16 is also implemented to check if the accuracy can be increased.

### Assignment Objective:
1. Load the CIFAR-10 dataset.
2. Preprocess the data (e.g. grayscale, normalize, reshape).
3. Train a classifier on the data:
   - A logistic regression classifier.
   - A neural network classifier.
4. Save a classification report.
5. Save a plot of the loss curve during training (for the MLP Classifier in scikit-learn).

## Data
The used CIFAR-10 dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Steps to Run the Analysis

### Setting Up the Environment
1. **Run the setup:**
   
```bash
sh setup.sh
```

### Running the Code
The project is structured with scripts saved in the `src` folder and the output saved in the `out` folder.

#### Logistic Regression
To run the logistic regression script:

``` 
python src/logistic_regression.py
```

#### Neural Network
To run the neural network script:

```bash
python src/neural_network.py
```

#### VGG16
To run the VGG16 script:

```bash
python src/vgg16_classifier.py
```

## Summary of Results
### Logistic Regression
##### **Model Accuracy: 0.32**

##### **Classification Report:**

| Class      | Precision | Recall | F1-Score | Support |
|------------|------------|--------|----------|---------|
| Airplane   | 0.34       | 0.39   | 0.36     | 1000    |
| Automobile | 0.37       | 0.38   | 0.37     | 1000    |
| Bird       | 0.27       | 0.21   | 0.23     | 1000    |
| Cat        | 0.23       | 0.16   | 0.19     | 1000    |
| Deer       | 0.25       | 0.21   | 0.23     | 1000    |
| Dog        | 0.30       | 0.30   | 0.30     | 1000    |
| Frog       | 0.28       | 0.33   | 0.30     | 1000    |
| Horse      | 0.32       | 0.33   | 0.33     | 1000    |
| Ship       | 0.34       | 0.41   | 0.37     | 1000    |
| Truck      | 0.39       | 0.46   | 0.42     | 1000    |
| **Accuracy** |            |        | 0.32     | 10000   |
| **Macro Avg** | 0.31    | 0.32   | 0.31     | 10000   |
| **Weighted Avg** | 0.31 | 0.32   | 0.31     | 10000   |

### Neural Network
##### **Model Accuracy: 0.44**

##### **Classification Report:**

| Class      | Precision | Recall | F1-Score | Support |
|------------|------------|--------|----------|---------|
| Airplane   | 0.50       | 0.44   | 0.47     | 1000    |
| Automobile | 0.52       | 0.53   | 0.53     | 1000    |
| Bird       | 0.34       | 0.32   | 0.33     | 1000    |
| Cat        | 0.29       | 0.26   | 0.27     | 1000    |
| Deer       | 0.37       | 0.31   | 0.34     | 1000    |
| Dog        | 0.34       | 0.48   | 0.40     | 1000    |
| Frog       | 0.48       | 0.51   | 0.50     | 1000    |
| Horse      | 0.53       | 0.52   | 0.53     | 1000    |
| Ship       | 0.50       | 0.49   | 0.50     | 1000    |
| Truck      | 0.50       | 0.52   | 0.51     | 1000    |
| **Accuracy** |            |        | 0.44     | 10000   |
| **Macro Avg** | 0.44    | 0.44   | 0.44     | 10000   |
| **Weighted Avg** | 0.44 | 0.44   | 0.44     | 10000   |

### VGG16 (CNN)
Note: The VGG16 model implementation is an additional exploration and not the primary focus of the assignment.

#### **Model Accuracy:**

#### **Classification Report:**

## Discussion

### Steps Taken for Logistic and Neural Network Regression

**Logistic Regression:**
- **Preprocessing:** Images were converted to grayscale, normalized, and flattened.
- **Training:** The logistic regression model was trained using scikit-learn's LogisticRegression with hyperparameter tuning via GridSearchCV.
- **Evaluation:** A classification report and confusion matrix were generated to evaluate model performance.

**Neural Network:**
- **Preprocessing:** Similar preprocessing steps were taken as for logistic regression.
- **Training:** An MLPClassifier from scikit-learn was used with hyperparameter tuning. Early stopping and model checkpointing were implemented to optimize training.
- **Evaluation:** The model's performance was assessed using a classification report and confusion matrix.

### Considerations for the VGG16 Model
In implementing the VGG16 model, several considerations were taken into account, inspired by [this Kaggle notebook](https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16):

**Data Augmentation:** 
- Parameters for data augmentation included rotation, zoom, width shift, height shift, shear, and horizontal flip, all set to enhance the training dataset's diversity.

**Network Structure:** 
- The VGG16 model's convolutional base was used with additional Dense layers and Dropout layers to improve generalization and reduce overfitting.

**SGD Parameters:** 
- The model was trained using the SGD optimizer with a learning rate of 0.001 and a momentum of 0.9.

**Hyperparameters:** 
- The training was conducted over 100 epochs with a batch size of 128, following the approach detailed in the referenced Kaggle notebook.


## Discussion of Limitations and Possible Improvements

### Limitations

**1. Logistic Regression:**
- Limited capability to capture complex patterns in the data, which could explain the low accuracy of .32 for the classification.
- Looking at the heatmap..

**2. Neural Network:**
- Despite being more accurate than logistic regression, the simple MLP used here still struggles with the classification.
- The heatmap suggests...

### Improvements

- **Use of CNNs:** Convolutional Neural Networks (CNNs), like the VGG16 model, are better suited for image classification.
- **Data Augmentation:** Implementing data augmentation can help improve generalization by artificially increasing the size and variability of the training data.
