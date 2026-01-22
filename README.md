# How does music affect mental health ? 🎵

## Project overview

**Goal** : Explore and analyse the relationship between music listening habits and self reported mental health conditions and evaluate whether patterns in music preferences and listening habits have predictive power of determining the presence of Anxiety, Depression, OCD or Insomnia

**Dataset** : [mxmh-survey-results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)


## Problem Formulations 

**Aim** : Multi Label Classification

**Model used** : TabNetClassifier

**Evaluation Metrics** : Precision, Recall, F1-score, ROC-AUC, PR-AUC


## Notebook Structure and Methodology :

**Exploratory Data Analysis** :
   
 Ran basic statistical reports and plots , analysed the average mental health scores by music effects and the effects of number of listening hours on the same. 
 
**Data PreProcessing** :

1. Numerical features: Missing Values are imputed using median and then normalized
2. Categorical features: - Missing values are imputed using the most frequent category and categorical values are encoded using OrdinalEncoder
3. ColumnTransformer applies these transformations column-wise and is fit only on the training data to prevent data leakage


**Model detail** :

TabNetClassifier (PyTorch TabNet) is a deep learning architecture for tabular data which uses a sequential attention mechanism to select which features to reason from at each decision step, focusing the model's capacity on the most important features. Details [here](https://arxiv.org/abs/1908.07442)

**Model Architecture** : 

First I used the architecture as it is from PyTorch Tabnet's official docs and then trained the classifier model on each mental health condition and tested it on the test set. To handle class imbalance I used TabNet's built in ClassificationSMOTE augmentation which helps in interpolating the minor class by using its nearest neighbours.
Second I modified the hyperparameters after using OPTUNA study ( hyperparameter search framework based on Bayesian optimization For each mental health condition, TabNet architectural and optimization parameters are tuned independently using validation performance as the objective.


Detailed Summary of the Results can be found in the notebook. Feel free to fork a copy of your own and play around with the data [here](https://www.kaggle.com/code/debarupasinha/how-does-music-affects-mental-health-wip)

