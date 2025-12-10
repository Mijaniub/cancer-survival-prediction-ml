# cancer-survival-prediction-ml
End-to-end ML pipeline for predicting cancer patient survival using clinical, lifestyle, and demographic features.
🧬 Survival Prediction of Cancer Patients Using Machine Learning

This project builds an end-to-end machine learning pipeline to predict the survival outcome of cancer patients using demographic, clinical, lifestyle, and treatment-related features. The goal is to apply data preprocessing, feature engineering, model training, and evaluation to derive meaningful insights from clinical datasets.

📂 Project Overview

Cancer survival prediction plays a crucial role in early diagnosis and treatment planning.
This machine learning project:

Cleans and preprocesses raw patient data

Encodes categorical features

Handles class imbalance using SMOTE

Performs exploratory data analysis

Trains multiple ML models

Tunes XGBoost, the final selected model

Generates predictions for submission

📊 Dataset Description

Patient data in CSV format (Training + Testing)

~48 features including:

Demographic: Age, Gender, Country, Urban/Rural

Lifestyle: Smoking, Alcohol, Physical Activity, Diet

Clinical: Tumor size, Cancer stage, Diagnosis delay

Treatment: Treatment type, Transfusion, Screening

Target variable: Survival Prediction (0/1)

Challenges:

Many categorical columns

Missing values

High-cardinality fields

Imbalanced labels

🔧 Data Preprocessing

Label encoding for categorical variables

Extracted Age from Date of Birth

Imputed missing values (alcohol, screening)

Dropped irrelevant or noisy columns

Converted tumor size into bins

Added Geo-Area as a custom feature

Balanced classes using SMOTE

🧪 Exploratory Data Analysis (EDA)

Charts included in notebook:

Survival distribution

Tumor size vs. Survival

Treatment vs. Survival

Age distribution

Key findings:

Dataset is imbalanced (more survivors)

High tumor size reduces survival chance

Combination treatments show better survival

Most patients aged 40–70

🤖 Modeling

Models tested:

Logistic Regression

Decision Tree

Random Forest

XGBoost (Final Model)

Why XGBoost?

Works great with structured data

Handles missing values

Strong regularization

Reduces overfitting

Performs well with imbalanced classes

Hyperparameter Tuning

Used RandomizedSearchCV on:

n_estimators

max_depth

learning_rate

subsample

colsample_bytree

Metric used: F1 Score (weighted)
Cross-validation: 5-fold

📈 Model Performance
Metric	Score
Accuracy	~52%
Precision	~51%
Recall	~53%
F1 Score	~50%

The results indicate difficulty in predicting survival due to noise and imbalance in data.

📤 Submission

The final predictions were exported as:

Patient_ID,Survival Prediction
1001,1
1002,0
...

🧩 Challenges

Strong class imbalance

Potential label leakage

Limited numeric features

Difficulty encoding high-cardinality variables

📘 Tools & Libraries

Python 3.10+

pandas, numpy

matplotlib, seaborn

sklearn

xgboost

imblearn

Jupyter Notebook / Google Colab

🚀 Future Improvements

Use temporal data (treatment intervals)

More robust feature selection (SHAP, RFE)

Ensemble or stacked models

Cost-sensitive learning

Larger, richer dataset

📁 Suggested GitHub Repo Structure
cancer-survival-prediction/
│── data/
│   ├── train.csv
│   ├── test.csv
│── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│── models/
│   ├── xgboost_final.pkl
│── submissions/
│   ├── final_submission.csv
│── README.md
│── requirements.txt
│── final-report.pdf
