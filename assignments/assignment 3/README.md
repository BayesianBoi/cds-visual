# Assignment 3 - Document classification using pretrained image embeddings

## Repository overview

This project aims to classify documents based on their visual appearance using a pretrained VGG16 model. The documents are classified into categories such as advertisements, emails and forms. This task focuses on predicting the type of document based solely on its appearance rather than its contents.

### Assignment objectives
1. Load the Tobacco3482 data and generate labels for each image.
2. Train a classifier to predict document type based on visual features.
3. Present a classification report and learning curves for the trained classifier.
4. Include a short description of what the classification report and learning curve show.

## Data source

The dataset used is a subset of the Tobacco3482 dataset. It contains images of 10 different document types. The subset can be found and downloaded from [Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). The full Tobacco3482 dataset can be accessed from the original study [here](https://dl.acm.org/doi/abs/10.1145/1148170.1148307).


## Steps for running the analysis

### Setting up the environment
1. **Set up the virtual environment and install requirements:**
    ```bash
    sh setup.sh
    ```
2. **Activate the virtual environment:**
    ```bash
    source envVis3/bin/activate
    ```

### Running the code
1. **[Download](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download) and place the data set in the `/in` folder**

2. **Run the main analysis script:**
    ```bash
    python src/main_analysis.py
    ```
This script will load the data, train the model and evaluate the model.

## Summary of results
### Classification report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| ADVE         | 0.92      | 0.98   | 0.95     | 46      |
| Email        | 0.95      | 0.96   | 0.95     | 120     |
| Form         | 0.81      | 0.90   | 0.85     | 86      |
| Letter       | 0.88      | 0.88   | 0.88     | 114     |
| Memo         | 0.86      | 0.86   | 0.86     | 124     |
| News         | 0.91      | 0.84   | 0.88     | 38      |
| Note         | 0.67      | 0.82   | 0.74     | 40      |
| Report       | 0.66      | 0.62   | 0.64     | 53      |
| Resume       | 0.83      | 0.79   | 0.81     | 24      |
| Scientific   | 0.61      | 0.44   | 0.51     | 52      |
| **Accuracy** |           |        | 0.84     | 697     |
| **Macro avg**| 0.81      | 0.81   | 0.81     | 697     |
| **Weighted avg** | 0.83  | 0.84   | 0.83     | 697     |

### Learning curves
![Learning Curves](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%203/out/loss_curve.png)

### Confusion matrix
![Confusion Matrix](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%203/out/confusion_matrix.png)

## Discussion
### Key points
The model achieved an overall accuracy of 84%, which is good given the task of document classification based on visual appearance alone. I have seen other multi-modal approaches that reached a similar accuracy. The confusion matrix indicates that most classes are predicted well.

### Loss curve
The training and validation loss curves show that the model converged well. Initially, the validation loss decreased alongside the training loss, which is a good indication that the model was learning effectively. After a few epochs, the validation loss began to stabilize, suggesting that the model had learned the important features for classification and was no longer improving significantly. There is no significant sign of overfitting as the validation loss does not increase substantially.

## Limitations and possible steps
### Limitations 
- **Class Imbalance:** The data set is in generally quite imbalanced. Some classes have fewer examples which might affect the model's performance.

### Possible improvements
- **Further Fine-Tuning:** More fine-tuning of the pretrained VGG16 model could potentially improve accuracy. I have tried countless different setups and this was the one that reached the highest accuracy. Similarly to further tweaking the model, changing parameters such as learning rate, batch size, and dropout rate might give better results. Also been tweaking around with them and the current ones were the ones that gave the best results.
- **Additional Data:** Given the model the full data set (rather than a subset) would most likely increase the accuracy (as the subset is quite imbalanced in terms of how well the different categories of documents are represented)
- **Multi-modal Approaches:** Combining the image data analysed here with a language model could yield better accuracy. As the visual representation of many of the document types are similar, the language model using the contents of the documents could better help distinguish them.
