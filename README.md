💳 Fraudulant_transaction_Detection

A supervised machine learning project for detecting fraudulent credit card transactions using Logistic Regression and Random Forest on a highly imbalanced dataset.

The project focuses on building a reliable fraud detection pipeline using SMOTE, Scikit-learn/imblearn pipelines, stratified splitting, and hyperparameter tuning with GridSearchCV.

📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small proportion of all transactions.

A model that simply predicts every transaction as legitimate can achieve very high accuracy while failing to detect fraud. Therefore, this project focuses on metrics that are more meaningful for fraud detection:

Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion Matrix

The project compares Logistic Regression and Random Forest, handles class imbalance using SMOTE, and uses GridSearchCV for hyperparameter optimization.

🎯 Project Objective

The main objective is to develop a machine learning pipeline capable of identifying fraudulent credit card transactions while maintaining a good balance between:

Detecting fraudulent transactions and minimizing false fraud alerts.

The final selected model is a tuned Random Forest Classifier with a classification threshold of 0.60.

📊 Dataset

The project uses the Credit Card Fraud Detection dataset provided by the Machine Learning Group at Université Libre de Bruxelles (ULB).

The dataset contains anonymized credit card transactions with:

Time
V1 through V28
Amount
Class
Target Variable
Class	Meaning
0	Legitimate transaction
1	Fraudulent transaction

The dataset is highly imbalanced, making conventional accuracy an unsuitable primary evaluation metric.

Dataset Size

The original CSV file is approximately 149 MB, so it is not included in this GitHub repository.

See data/README.md for dataset download and setup instructions.

🧠 Machine Learning Architecture

                         RAW DATA
                            │
                            ▼
                  Data Understanding
                            │
                            ▼
                           EDA
                            │
                            ▼
                    Data Preprocessing
                            │
                            ▼
                Stratified Train/Test Split
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
        Logistic Regression      Random Forest
                 │                     │
          StandardScaler               │
                 │                     │
               SMOTE                 SMOTE
                 │                     │
                 ▼                     ▼
        Logistic Regression      Random Forest
                 │                     │
                 └──────────┬──────────┘
                            ▼
                      GridSearchCV
                            │
                            ▼
                   Best Hyperparameters
                            │
                            ▼
                  Threshold Optimization
                            │
                            ▼
                   Final Test Evaluation
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      Precision           Recall           ROC-AUC
                            │
                            ▼
                         PR-AUC

🔎 Exploratory Data Analysis

The EDA stage investigates the structure, distributions, imbalance, and relationships within the dataset.

EDA includes:

Dataset dimensions
Data types
Missing-value analysis
Duplicate detection
Class distribution
Transaction amount distribution
Time-based analysis
V1-V28 distributions
Amount vs Class
Hour vs Class
V-features vs Class
Correlation heatmap
PCA visualization
Important EDA observations
The dataset is highly imbalanced.
Fraudulent transactions represent a very small minority.
Amount has a strongly right-skewed distribution.
Several PCA-derived V-features show noticeable differences between legitimate and fraudulent transactions.
Correlation analysis helps identify relationships between numerical features.
PCA-derived V-features generally have low inter-feature correlation, which is expected from PCA-transformed components.

🧹 Data Preprocessing

The preprocessing workflow includes:


                Raw Dataset
                      ↓
             Data Quality Checks
                      ↓
            Duplicate Detection
                      ↓
              Duplicate Removal
                      ↓
            Feature Engineering
                      ↓
        Feature / Target Separation
                      ↓
        Stratified Train/Test Split 

Exact duplicate rows are investigated and removed before model training.

The dataset is divided using a stratified train/test split to preserve the minority-class proportion in both subsets.

⚖️ Handling Class Imbalance

Because fraud transactions are extremely rare, class imbalance is handled using:

SMOTE

Synthetic Minority Over-sampling Technique

SMOTE generates synthetic minority-class samples to provide the model with more representative fraud examples during training.

Importantly, SMOTE is implemented inside the imbalanced-learn pipeline.

    Logistic Regression

                    StandardScaler
                          ↓
                        SMOTE
                          ↓
                    Logistic Regression
    Random Forest
                        SMOTE
                          ↓
                        Random Forest

SMOTE is therefore applied separately within the training folds during cross-validation, helping prevent data leakage.

🤖 Models

Two supervised classification algorithms were evaluated.

1. Logistic Regression

Logistic Regression provides a strong linear baseline for binary classification.

Pipeline:

                    StandardScaler
                          ↓
                        SMOTE
                          ↓
                    Logistic Regression
2. Random Forest

Random Forest is an ensemble tree-based model capable of learning nonlinear relationships and interactions between features.

Pipeline:

                        SMOTE
                          ↓
                    Random Forest

Feature scaling is not required for Random Forest.

🔧 Hyperparameter Tuning

GridSearchCV with cross-validation is used to optimize the model hyperparameters.

The tuning process searches through predefined parameter combinations and selects the configuration providing the best cross-validation performance.

The tuned models are then evaluated on the untouched test set.

📏 Evaluation Metrics

Because this is a highly imbalanced fraud detection problem, Accuracy is not used as the primary evaluation metric.

The main evaluation metrics are:

Precision

Measures how many transactions predicted as fraud were actually fraudulent.

Precision = TP / (TP + FP)
Recall

Measures how many actual fraudulent transactions were successfully detected.

Recall = TP / (TP + FN)
F1-score

Harmonic mean of Precision and Recall.

                    F1 = 2 × (Precision × Recall)
                         -------------------------
                           Precision + Recall
ROC-AUC

Measures the model's ability to distinguish between legitimate and fraudulent transactions across classification thresholds.

PR-AUC

Measures the quality of the precision-recall trade-off and is particularly useful for highly imbalanced classification problems.

🏆 Model Comparison

The evaluated models produced the following results.

Model	Precision	Recall	F1	ROC-AUC	PR-AUC
Logistic Regression	5.21%	87.37%	9.83%	95.97%	68.25%
Random Forest	93.59%	76.84%	84.39%	95.50%	81.20%
Tuned Logistic Regression	5.21%	87.37%	9.83%	95.97%	68.25%
Tuned Random Forest	91.25%	76.84%	83.43%	96.57%	82.49%

🎯 Final Model

Tuned Random Forest

The tuned Random Forest achieved the strongest overall ranking performance:

ROC-AUC = 0.96565

PR-AUC  = 0.82492

At the default classification threshold of 0.50:

Precision = 91.25%

Recall    = 76.84%

F1-score  = 83.43%

However, because fraud detection also requires minimizing false-positive alerts, an additional threshold analysis was performed.

🎚️ Threshold Optimization

The default classification threshold is:

0.50

The tuned Random Forest was also evaluated at:

0.60

Results at threshold = 0.60

Precision = 97.30%

Recall    = 75.79%

F1-score  = 85.21%

ROC-AUC   = 96.57%

PR-AUC    = 82.49%

Confusion Matrix

                                 Predicted
                                 Legit   Fraud
                Actual Legit     56649      2
                Actual Fraud        23     72

Therefore:

True Negatives  = 56,649

False Positives = 2

False Negatives = 23

True Positives  = 72

Threshold comparison

                    Metric	Threshold 0.50	Threshold 0.60
                    Precision   91.25%	    97.30%
                    Recall	    76.84%	    75.79%
                    F1-score	83.43%	    85.21%
                    ROC-AUC	    96.57%	    96.57%
                    PR-AUC	    82.49%	    82.49%
                    False Positives	7	       2
                    False Negatives	22	      23

Why threshold 0.60?

Increasing the threshold makes the classifier more conservative when labeling a transaction as fraudulent.

The change from 0.50 to 0.60:

Reduces false positives from 7 to 2
Increases Precision from 91.25% to 97.30%
Reduces Recall slightly from 76.84% to 75.79%
Improves F1-score from 83.43% to 85.21%

Therefore, 0.60 provides a better precision-oriented operating point for this project.

Note: For a completely unbiased final threshold-selection procedure, the threshold should ideally be selected using validation/CV data and evaluated only once on the untouched test set.

🏅 Final Model Selection

Based on the experiments, the Tuned Random Forest Classifier is selected as the final model.

At a threshold of 0.60, it achieves:

                    FINAL MODEL


              Tuned Random Forest
                       │
                       ▼
              Threshold = 0.60
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Precision        Recall        F1-score
      97.30%         75.79%          85.21%
        │
        ├──────────────┐
        ▼              ▼
     ROC-AUC         PR-AUC
      96.57%          82.49%

The model detects the majority of fraudulent transactions while producing only 2 false-positive alerts on the test set.

📁 Project Structure

                credit-card-fraud-detection/
                │
                ├── README.md
                ├── requirements.txt
                ├── .gitignore
                │
                ├── data/
                │   └── README.md
                │
                ├── notebooks/
                │   ├── 01_preprocessing.ipynb
                │   ├── 02_eda.ipynb
                │   ├── 03_trainin_model.ipynb
                │
                ├── models/
                │   └── fraud_detection_model.pkl
                │
                ├── reports/
                │   ├── figures/
                │   └── results/
                │       └── model_comparison.csv
                │
                └── app/
                    └── app.py

⚙️ Installation

Clone the repository:

git clone <your-repository-url>

Move into the project directory:

cd credit-card-fraud-detection

Create and activate a virtual environment:

python -m venv .venv
Windows
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
📥 Dataset Setup

The dataset is approximately 149 MB and is not included in this repository.

Download the dataset from Kaggle and place it at:

data/creditcard.csv

See:

data/README.md

for detailed dataset instructions.

▶️ Running the Project

Start with the notebooks in the following order:

                    01_preprocessing.ipynb
                            ↓
                    02_eda.ipynb
                            ↓
                    03_trainin_model.ipynb

This order follows the complete machine learning workflow from data understanding to final evaluation.

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

imbalanced-learn

Matplotlib

Seaborn

Jupyter Notebook

Joblib

GridSearchCV

SMOTE

📚 Key Machine Learning Concepts

This project demonstrates:

Supervised Learning

Binary Classification

Exploratory Data Analysis

Data Preprocessing

Stratified Train/Test Split

Class Imbalance

SMOTE

Data Leakage Prevention

Machine Learning Pipelines

Logistic Regression

Random Forest

Cross-Validation

GridSearchCV

Hyperparameter Tuning

Classification Threshold Optimization

Precision

Recall

F1-score

ROC-AUC

PR-AUC

Confusion Matrix

🚀 Future Improvements

Possible improvements include:

Threshold selection using a dedicated validation set
Precision-Recall curve optimization
Testing additional imbalance-handling techniques
Cost-sensitive learning
XGBoost or other boosting algorithms
Feature importance analysis
Model explainability using SHAP
Streamlit deployment
Model monitoring
Real-time fraud prediction API

👨‍💻 Author

Arsalan Saqib

Bachelor of Computer Science

This project was developed as part of a supervised machine learning project focused on fraud detection.

⭐ Project Summary

                            Highly Imbalanced Dataset
                                      ↓
                                     EDA
                                      ↓
                            Data Preprocessing
                                      ↓
                            Stratified Split
                                      ↓
                                   SMOTE
                                      ↓
                             ┌────────┴────────┐
                             ▼                 ▼
                            Logistic         Random
                            Regression       Forest
                             │                 │
                             └────────┬────────┘
                                      ▼
                                GridSearchCV
                                      ↓
                              Tuned Random Forest
                                      ↓
                             Threshold = 0.60
                                      ↓
                             Precision = 97.30%
                             Recall    = 75.79%
                             F1        = 85.21%
                             ROC-AUC   = 96.57%
                             PR-AUC    = 82.49%

Final selected model: Tuned Random Forest Classifier with a 0.60 decision threshold.
