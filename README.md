# CreditWise Loan System

A supervised learning project that predicts whether a loan application will be approved based on applicant and financial details.

## Overview
This project applies classification models to a loan approval dataset to predict the `Loan_Approved` outcome (Yes/No). It covers the full pipeline: data cleaning, exploratory data analysis, feature engineering, and model comparison.

## Dataset
The dataset (`loan_approval_data.csv`) is excluded from this repository(see `.gitignore`). It includes features such as Applicant Income, Coapplicant Income, Age, Dependents, Credit Score, Existing Loans, DTI Ratio, Collateral Value, Loan Amount, Loan Term, Gender, Education Level, Loan Purpose, Employer Category, and Property Area.

## Steps Performed
- **Data Cleaning**: Handled missing values — mean imputation for numerical columns, most-frequent (mode) imputation for categorical columns.
- **Exploratory Data Analysis (EDA)**: Visualized class balance, categorical distributions (Gender, Education Level, Loan Purpose, Property Area, Employer Category), income distributions, and outliers across key features using box plots.
- **Feature Engineering**: Added squared terms for `DTI_Ratio` and `Credit_Score` to capture non-linear relationships; dropped the original columns after transformation.
- **Preprocessing**: Split data into train/test sets (80/20) and applied `StandardScaler` for feature scaling.

## Models Trained
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)

Each model was evaluated using Precision, Recall, F1 Score, Accuracy, and Confusion Matrix.

## Result
Based on precision, **Naive Bayes** performed best among the three models tested.

## Tech Stack
Python, pandas, numpy, seaborn, matplotlib, scikit-learn

## How to Run
1. Clone this repository
2. Place `loan_approval_data.csv` in the project folder (not included — see Dataset section)
3. Install dependencies: `pip install pandas numpy seaborn matplotlib scikit-learn`
4. Run `code.ipynb` in Jupyter Notebook or VS Code