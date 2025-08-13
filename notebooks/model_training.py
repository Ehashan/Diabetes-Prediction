import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load Data
df = pd.read_csv('data/diabetes.csv')

# Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Random Forest
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(X_train, y_train)

# Model Evaluation
# Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Random Forest
y_pred_rand_forest = rand_forest.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rand_forest))
print(confusion_matrix(y_test, y_pred_rand_forest))
print(classification_report(y_test, y_pred_rand_forest))

# Save Models
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rand_forest, f)

print("Models saved successfully.")
