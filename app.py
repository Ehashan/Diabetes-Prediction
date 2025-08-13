import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load Models
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
except FileNotFoundError:
    log_reg = None

try:
    with open('random_forest_model.pkl', 'rb') as f:
        rand_forest = pickle.load(f)
except FileNotFoundError:
    rand_forest = None

# Load Data
df = pd.read_csv('data/diabetes.csv')

# Sidebar
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "Data Exploration"

def set_page(page_name):
    st.session_state.page = page_name

with st.sidebar:
    if st.session_state.page == "Data Exploration":
        st.button("Data Exploration", on_click=set_page, args=("Data Exploration",), type="primary", use_container_width=True)
    else:
        st.button("Data Exploration", on_click=set_page, args=("Data Exploration",), use_container_width=True)

    if st.session_state.page == "Model Prediction":
        st.button("Model Prediction", on_click=set_page, args=("Model Prediction",), type="primary", use_container_width=True)
    else:
        st.button("Model Prediction", on_click=set_page, args=("Model Prediction",), use_container_width=True)

    if st.session_state.page == "Model Performance":
        st.button("Model Performance", on_click=set_page, args=("Model Performance",), type="primary", use_container_width=True)
    else:
        st.button("Model Performance", on_click=set_page, args=("Model Performance",), use_container_width=True)

page = st.session_state.page

# App Title and Description
st.title("Diabetes Prediction App")
st.markdown("This app predicts whether a patient has diabetes based on several health metrics.")

if page == "Data Exploration":
    st.header("Data Exploration")
    st.markdown(
        """
        <style>
            .stTabs [data-baseweb="tab-list"] button {
                background-color: transparent;
                color: white;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #FFC300;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["Dataset Overview", "Visualizations"])

    with tab1:
        st.subheader("Dataset Overview")
        st.write("Shape of the dataset:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Data Types:", df.dtypes)
        st.subheader("Sample Data")
        st.write(df.head())

    with tab2:
        st.subheader("Visualizations")
        st.write("Count plot of Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Outcome', data=df, ax=ax, palette=['#FFC300', '#FF5733'])
        st.pyplot(fig)

        st.write("Histogram of all features")
        fig = plt.figure(figsize=(10, 8))
        for i, col in enumerate(df.columns):
            plt.subplot(3, 3, i + 1)
            df[col].hist(bins=20)
            plt.title(col)
            plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="YlOrRd")
        st.pyplot(fig)


elif page == "Model Prediction":
    st.header("Model Prediction")
    st.subheader("Enter Patient Data")

    pregnancies = st.slider("Pregnancies", 0, 20, 1, help="Number of times pregnant")
    glucose = st.slider("Glucose", 0, 200, 120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure = st.slider("Blood Pressure", 0, 130, 70, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20, help="Triceps skin fold thickness (mm)")
    insulin = st.slider("Insulin", 0, 900, 80, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.slider("BMI", 0.0, 70.0, 32.0, help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, help="Diabetes pedigree function")
    age = st.slider("Age", 0, 120, 30, help="Age (years)")

    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

    model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

    if st.button("Predict"):
        with st.spinner('Making prediction...'):
            if glucose == 0 or blood_pressure == 0 or skin_thickness == 0 or insulin == 0 or bmi == 0:
                st.warning("Some input values are zero. This might affect the prediction accuracy.")

            if model_choice == "Logistic Regression":
                if log_reg is not None:
                    prediction = log_reg.predict(input_data)
                    probability = log_reg.predict_proba(input_data)
                else:
                    st.error("Logistic Regression model not found. Please train the model first.")
                    prediction = None
            else:
                if rand_forest is not None:
                    prediction = rand_forest.predict(input_data)
                    probability = rand_forest.predict_proba(input_data)
                else:
                    st.error("Random Forest model not found. Please train the model first.")
                    prediction = None

            if prediction is not None:
                if prediction[0] == 1:
                    st.error("Prediction: Diabetic")
                else:
                    st.success("Prediction: Not Diabetic")

                st.write("Prediction Probability:", probability)


elif page == "Model Performance":
    st.header("Model Performance")
    st.markdown(
        """
        <style>
            .stTabs [data-baseweb="tab-list"] button {
                background-color: transparent;
                color: white;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #FFC300;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Logistic Regression", "Random Forest"])

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with tab1:
        st.subheader("Model Comparison")
        if log_reg is not None and rand_forest is not None:
            y_pred_log_reg = log_reg.predict(X_test)
            y_pred_rand_forest = rand_forest.predict(X_test)

            accuracy_log_reg = (y_pred_log_reg == y_test).mean()
            accuracy_rand_forest = (y_pred_rand_forest == y_test).mean()

            accuracies = pd.DataFrame({
                'Model': ['Logistic Regression', 'Random Forest'],
                'Accuracy': [accuracy_log_reg, accuracy_rand_forest]
            })

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Model', y='Accuracy', data=accuracies, ax=ax, palette=['#FFC300', '#FF5733'])
            st.pyplot(fig)

    with tab2:
        st.subheader("Logistic Regression Performance")
        if log_reg is not None:
            y_pred_log_reg = log_reg.predict(X_test)
            report = classification_report(y_test, y_pred_log_reg, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.text("Classification Report:")
            st.dataframe(df_report)
            st.text("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', ax=ax, cmap="YlOrRd")
            st.pyplot(fig)
        else:
            st.write("Model not trained yet.")

    with tab3:
        st.subheader("Random Forest Performance")
        if rand_forest is not None:
            y_pred_rand_forest = rand_forest.predict(X_test)
            report = classification_report(y_test, y_pred_rand_forest, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.text("Classification Report:")
            st.dataframe(df_report)
            st.text("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred_rand_forest), annot=True, fmt='d', ax=ax, cmap="YlOrRd")
            st.pyplot(fig)
        else:
            st.write("Model not trained yet.")
