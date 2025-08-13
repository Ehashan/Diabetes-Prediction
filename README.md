# Diabetes Prediction Web Application

This project is a machine learning application that predicts the likelihood of a patient having diabetes based on several medical attributes. The application is built with Streamlit and includes two different prediction models: Logistic Regression and Random Forest.

## Project Structure

```
your-project/
├── app.py
├── requirements.txt
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── model_training.ipynb
└── README.md
```

## Dataset

The dataset used for this project is the **Pima Indians Diabetes Database** from Kaggle. It contains information about female patients of at least 21 years old of Pima Indian heritage.

- **Source:** [https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Models

Two machine learning models were trained for this prediction task:

1.  **Logistic Regression:** A linear model for classification.
2.  **Random Forest:** An ensemble learning method that operates by constructing a multitude of decision trees.

The model training process, including data exploration and evaluation, can be found in the `notebooks/model_training.ipynb` Jupyter Notebook.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-project.git
    cd your-project
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Application Features

The Streamlit application is organized into three main sections:

1.  **Data Exploration:**
    -   Displays an overview of the dataset, including its shape, columns, and data types.
    -   Shows a sample of the data.
    -   Provides interactive options for filtering the data.

2.  **Model Prediction:**
    -   Allows users to input patient data through various widgets.
    -   Provides real-time predictions from either the Logistic Regression or Random Forest model.
    -   Displays the prediction probability.

3.  **Model Performance:**
    -   (To be implemented) This section will display the evaluation metrics and performance charts for the trained models.

## Deployment

This application is ready to be deployed to Streamlit Cloud. To do so, connect your GitHub repository to your Streamlit Cloud account and follow the deployment instructions.
