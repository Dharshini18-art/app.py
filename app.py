import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode non-target categorical columns
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

# Load data
df = load_data()

# Title
st.title("📊 Telco Customer Churn Prediction")
st.write("This app uses a Logistic Regression model to predict churn based on customer data.")

# Data exploration
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Visualizations
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax1)
ax1.set_title("Customer Churn Count")
st.pyplot(fig1)

# Heatmap
st.subheader("Feature Correlation Heatmap")
df_numeric = df.select_dtypes(include=['number'])
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Train model
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = X_train.select_dtypes(include=['int64']).columns
numerical_features = X_train.select_dtypes(include=['float64']).columns

numerical_pipeline = Pipeline([('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
st.subheader("Model Performance")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig3, ax3 = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax3)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig3)

# Simple Prediction UI
st.subheader("Try a Sample Prediction")
sample_input = {}

for col in X.columns:
    unique_vals = df[col].unique()
    if len(unique_vals) <= 10:
        sample_input[col] = st.selectbox(f"{col}", sorted(unique_vals))
    else:
        sample_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([sample_input])

# Ensure input matches training features
for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

# Predict and show result
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
