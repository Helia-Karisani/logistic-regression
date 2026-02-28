# Logistic Regression for Customer Churn Prediction

## Overview

This project implements a **binary classification model** using **logistic regression** to predict **customer churn** (whether a customer leaves or stays).  
The notebook walks through data preprocessing, feature scaling, model training, probabilistic prediction, evaluation using **log loss**, and analysis of feature impact.

---

## Dataset

The dataset contains customer attributes and a binary churn label:

### Features
- `tenure`   – number of months the customer has stayed
- `age`
- `address`
- `income`
- `ed`       – education level
- `employ`   – years of employment
- `equip`    – equipment-related variable
- `callcard` – call card ownership indicator

### Target
- `churn`  
  - `0` = customer stays  
  - `1` = customer churns  

---

## Data Preprocessing

Steps performed:
1. Selected relevant numerical features
2. Converted the target variable (`churn`) to integer
3. Converted features and labels to NumPy arrays
4. Applied **standardization** using `StandardScaler`
5. Split data into training and test sets (80/20 split)

Feature scaling is essential because logistic regression is sensitive to feature magnitude.

---

## Logistic Regression Model

### Mathematical Formulation

Logistic regression models the **probability** that a data point belongs to class 1:

\[
P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

where

\[
z = \mathbf{w}^\top \mathbf{x} + b
\]

- \(\mathbf{x}\) is the feature vector
- \(\mathbf{w}\) are the learned coefficients
- \(b\) is the intercept
- \(\sigma(\cdot)\) is the sigmoid function

The output is a probability in \([0,1]\).

---

### Decision Rule

The predicted class is obtained by thresholding the probability:

\[
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1 \mid x) \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

---

## Probability Prediction

The model uses:

- `predict()` → predicted class labels (0 or 1)
- `predict_proba()` → class probabilities

Important detail:
- The columns of `predict_proba` correspond to `model.classes_`
- Column order must **never be assumed**

---

## Model Evaluation: Log Loss

### Log Loss Definition

Log loss (cross-entropy loss) penalizes incorrect and **overconfident** predictions:

\[
\text{LogLoss} = -\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
\]

Where:
- \(y_i \in \{0,1\}\) is the true label
- \(p_i\) is the predicted probability for class 1

Lower log loss indicates:
- Better probability calibration
- Higher confidence on correct predictions
- Stronger probabilistic modeling

---

## Feature Expansion Analysis (`callcard`)

A second model was trained after **adding the `callcard` feature**.

### Observation
- The log loss **decreased** after adding `callcard`

### Interpretation
- `callcard` provides **additional signal**
- The model produces probabilities closer to the true outcomes
- Improvement is probabilistic, not necessarily accuracy-based

This demonstrates that:
- Accuracy alone is insufficient
- Log loss captures improvements in confidence and calibration

---

## Coefficient Interpretation

In logistic regression:
- Positive coefficient → increases churn probability
- Negative coefficient → decreases churn probability
- Magnitude reflects strength of influence (after scaling)

Coefficients operate in **log-odds space**, not probability space.

---

## Key Takeaways

- Logistic regression is a **probabilistic model**, not just a classifier
- `predict_proba` must be interpreted using `model.classes_`
- Feature scaling is mandatory for meaningful coefficients
- Log loss is a stronger metric than accuracy for churn prediction
- Adding informative features improves probability calibration

---

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Jupyter Notebook

---

## Final Note

This notebook emphasizes **correct modeling practices**, careful evaluation, and debugging awareness — particularly around probability outputs and feature pipelines.
