# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## Repository overview
This repository contains scripts to classify the CIFAR-10 dataset using Logistic Regression and Neural Networks. Additionally, a CNN model using VGG16 is also implemented to check if the accuracy can be increased.

### Assignment objective:
1. Load the Cifar10 dataset
2. Preprocess the data (e.g. greyscale, normalize, reshape)
3. Train a classifier on the data
   - A logistic regression classifier and a neural network classifier
4. Save a classification report
5. Save a plot of the loss curve during training (only for the MLP Classifier)

## Data source
The used CIFAR-10 dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Steps for running the analysis

### Setting up the environment
1. **Set up the virtual environment and install requirements:**
    ```bash
    sh setup.sh
    ```
2. **Activate the virtual environment:**
    ```bash
    source envVis2/bin/activate
    ```

### Running the Code

#### Logistic Regression
- **To run the logistic regression script:**

```bash
python src/logistic_regression.py
```

#### Neural Network
- **To run the neural network script:**

```bash
python src/neural_network.py
```

#### VGG16
- **To run the VGG16 script:**

```bash
python src/vgg16_classifier.py
```

## Summary of results
### Logistic Regression
##### **Model accuracy: 0.32**

##### **Classification report:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|------------|--------|----------|---------|
| Airplane    | 0.34       | 0.39   | 0.36     | 1000    |
| Automobile  | 0.37       | 0.38   | 0.37     | 1000    |
| Bird        | 0.27       | 0.21   | 0.23     | 1000    |
| Cat         | 0.23       | 0.16   | 0.19     | 1000    |
| Deer        | 0.25       | 0.21   | 0.23     | 1000    |
| Dog         | 0.30       | 0.30   | 0.30     | 1000    |
| Frog        | 0.28       | 0.33   | 0.30     | 1000    |
| Horse       | 0.32       | 0.33   | 0.33     | 1000    |
| Ship        | 0.34       | 0.41   | 0.37     | 1000    |
| Truck       | 0.39       | 0.46   | 0.42     | 1000    |
| **Accuracy**|            |        | 0.32     | 10000   |
| **Macro Avg** | 0.31    | 0.32   | 0.31     | 10000   |
| **Weighted Avg** | 0.31 | 0.32   | 0.31     | 10000   |


#### Confusion matrix (LR):
![Logistic Regression Confusion Matrix](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%202/out/cm_logistic.png)

### Neural Network
##### **Model accuracy: 0.43**

##### **Classification report:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|------------|--------|----------|---------|
| Airplane    | 0.45       | 0.49   | 0.47     | 1000    |
| Automobile  | 0.54       | 0.45   | 0.49     | 1000    |
| Bird        | 0.35       | 0.30   | 0.33     | 1000    |
| Cat         | 0.30       | 0.25   | 0.27     | 1000    |
| Deer        | 0.36       | 0.34   | 0.35     | 1000    |
| Dog         | 0.38       | 0.40   | 0.39     | 1000    |
| Frog        | 0.44       | 0.48   | 0.46     | 1000    |
| Horse       | 0.48       | 0.52   | 0.49     | 1000    |
| Ship        | 0.51       | 0.51   | 0.51     | 1000    |
| Truck       | 0.47       | 0.56   | 0.51     | 1000    |
| **Accuracy**|            |        | 0.43     | 10000   |
| **Macro Avg** | 0.43    | 0.43   | 0.43     | 10000   |
| **Weighted Avg** | 0.43 | 0.43   | 0.43     | 10000   |


#### Confusion matrix (NN):
![Neural Network Confusion Matrix](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%202/out/cm_nn.png)

#### Loss curve (NN)
![Neural Network Loss Curve](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%202/out/loss_curve_nn.png)

### VGG16 (CNN)
Note: The VGG16 model implementation is additional exploration and not the primary focus of the assignment

#### **Model accuracy: 0.64**

#### **Classification report:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|------------|--------|----------|---------|
| Airplane    | 0.70       | 0.69   | 0.69     | 1000    |
| Automobile  | 0.70       | 0.75   | 0.73     | 1000    |
| Bird        | 0.64       | 0.44   | 0.52     | 1000    |
| Cat         | 0.55       | 0.41   | 0.47     | 1000    |
| Deer        | 0.58       | 0.48   | 0.53     | 1000    |
| Dog         | 0.59       | 0.65   | 0.62     | 1000    |
| Frog        | 0.55       | 0.81   | 0.66     | 1000    |
| Horse       | 0.68       | 0.68   | 0.68     | 1000    |
| Ship        | 0.75       | 0.74   | 0.75     | 1000    |
| Truck       | 0.67       | 0.74   | 0.70     | 1000    |
| **Accuracy**|            |        | 0.64     | 10000   |
| **Macro Avg** | 0.64    | 0.64   | 0.63     | 10000   |
| **Weighted Avg** | 0.64 | 0.64   | 0.63     | 10000   |


#### Confusion matrix (VGG16):
![VGG16 Confusion Matrix](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%202/out/cm_vgg16.png)


## Discussion
### Steps Taken for Logistic and Neural Network Regression

### Logistic Regression:
- **Preprocessing:** Images were converted to grayscale, normalized and flattened
- **Training:** The logistic regression model was trained using scikit-learn with hyperparameter tuning via GridSearch
- **Evaluation:** A classification report and confusion matrix were generated to evaluate model performance

### Neural Network:
- **Preprocessing:** Similar preprocessing steps were taken as for logistic regression.
- **Training:** An MLPClassifier from scikit-learn was used with hyperparameter tuning. Early stopping and model checkpointing were implemented to optimize training
- **Evaluation:** The model's performance was assessed using a classification report, loss-curve and confusion matrix

### Considerations for the VGG16 Model
The VGG16 model was inspired by [this Kaggle notebook](https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16). I implemented data augmentation, a similar network structure with additional dense layers and dropout layers, the same SGD parameters with a learning rate of 0.001 and a momentum of 0.9, and the same epoch and batch size of 100 and 128, respectively. However, despite using an almost identical setup to theirs, my VGG16 model only reached an accuracy of .64 compared to their accuracy of 0.89

## Limitations and Possible Improvements

### Limitations
**1. Logistic Regression:**
- LR generally is quite bad at capturing complex patterns in the data, which could explain the low accuracy of 0.32
- Looking at the heatmap, the logistic regression struggles with classes that are similar in representation such as cats and dogs

**2. Neural Network:**
- Despite being more accurate than logistic regression, the MLP used here still struggles with the classification at an accuracy of just below chance 0.43
- Similar to the logistic regression, the heatmap also suggests misclassifications between similar classes

### Possible improvements
- **Use of CNNs:** Using CNNs like the used VGG16 model might be better suited for image classification. Further tuning the VGG16 model could have yielded higher accuracy (as shown by the kaggle notebook used for inspiration)
