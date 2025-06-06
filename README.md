# House Price Prediction using ElasticNet Regression

## Overview
This project focuses on predicting house prices based on various features using Elastic Net Regression, a powerful regularized linear regression technique that combines both L1 (Lasso) and L2 (Ridge) penalties. The goal is to build a model that can accurately estimate house prices by balancing feature selection and coefficient shrinkage to improve prediction performance and avoid overfitting.

## Dataset
The dataset contains housing information such as:
1. Location details (e.g., neighborhood, city)
2. Property characteristics (e.g., size in square feet, number of bedrooms/bathrooms, age)
3. Other relevant features (e.g., proximity to amenities, lot size, condition)

The dataset was obtained from Kaggle, at this link: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

## Methodology 

### Why Elastic Net Regression?
Elastic Net combines the strengths of Ridge and Lasso regression:
1. L1 penalty for feature selection (sparse model)
2. L2 penalty for coefficient shrinkage (handles multicollinearity)
This helps improve generalization and model interpretability on datasets with many correlated features.

### Model Preprocessing, Training and Evaluation
1. Outliers and Missing Data was dealt with, by removal of rows, and imputation of data.
2. The categorical data was dealt with by using dummy variables.
3. The data was first split into features, and target, which is the ```SalePrice``` columns.
4. The data was further split into train and test sets using ```scikit-learn```'s ```train_test_split```.
5. The data was scaled using ```StandardScaler```.
6. Multiple hyperparameters, ```alpha``` and ```l1ratio``` were used, to train the model and choose the best model using GridSearchCV.
7. The model was evaluated using RMSE and Mean Absolute error.

## Results Obtained

The root mean squared error of the model was 20619.58, and the mean absolute error was obtained as 14218.35, which is almost 10% as compared to the mean sale price of 180815.53. 

The model was saved as a .pkl file, available in the models folder, and was explained through SHAP. 
