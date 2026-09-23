# Logistic Regression for Customer Churn Prediction

## Overview

This project uses **logistic regression** to predict **customer churn** (whether a customer leaves or stays).
The notebook covers data preprocessing, feature scaling, model training, probability prediction, evaluation with **log loss**, and the effect of individual features.

---

## Dataset

### Features
- `tenure`: number of months the customer has stayed
- `age`
- `address`
- `income`
- `ed`: education level
- `employ`: years of employment
- `equip`: equipment-related variable
- `callcard`: call card ownership

### Target
- `churn`: `0` = customer stays, `1` = customer churns

---

## Data Preprocessing

1. Selected the numerical features
2. Converted `churn` to integer
3. Converted features and labels to NumPy arrays
4. Standardized features with `StandardScaler`
5. Split data into training and test sets (80/20)

Scaling matters because logistic regression is sensitive to feature magnitude.

---

## Logistic Regression

The model estimates the probability that a sample belongs to class 1:

```
P(y = 1 | x) = 1 / (1 + exp(-(w^T x + b)))
```

- x is the feature vector
- w are the learned coefficients
- b is the intercept

The predicted class is 1 if `P(y = 1 | x) >= 0.5`, otherwise 0.

---

## Probability Prediction

- `predict()` returns class labels (0 or 1)
- `predict_proba()` returns class probabilities

The columns of `predict_proba` follow `model.classes_`, so the column order should be checked rather than assumed.

---

## Evaluation: Log Loss

```
LogLoss = -(1 / N) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

- y_i is the true label
- p_i is the predicted probability for class 1

Lower log loss means better calibrated probabilities. It also penalizes predictions that are confident and wrong.

---

## Adding the `callcard` Feature

A second model was trained after adding `callcard`. Log loss decreased, so `callcard` adds useful signal and the predicted probabilities move closer to the true outcomes. The improvement shows up in log loss even when accuracy barely changes.

---

## Coefficients

- Positive coefficient: increases churn probability
- Negative coefficient: decreases churn probability
- After scaling, the magnitude shows the strength of the effect

Coefficients are in log-odds space, not probability space.

---

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Jupyter Notebook
