import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset safely
@st.cache_data
def load_data():
    file_path = "telecommunications_churn(1).csv"
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: `{file_path}`. Please upload it or check your repo.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"⚠️ Error reading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# Proceed only if data is available
if df.empty:
    st.stop()

# Feature Engineering
df['avg_day_call_duration'] = df['day_mins'] / (df['day_calls'] + 1e-5)
df['avg_evening_call_duration'] = df['evening_mins'] / (df['evening_calls'] + 1e-5)
df['avg_night_call_duration'] = df['night_mins'] / (df['night_calls'] + 1e-5)
df['avg_international_call_duration'] = df['international_mins'] / (df['international_calls'] + 1e-5)
df['total_calls'] = df['day_calls'] + df['evening_calls'] + df['night_calls'] + df['international_calls']
df['total_mins'] = df['day_mins'] + df['evening_mins'] + df['night_mins'] + df['international_mins']
df['plan_combination'] = df['international_plan'].astype(str) + "_" + df['voice_mail_plan'].astype(str)

# Target and features
target = df['churn']
features = df.drop(['churn'], axis=1)

# Preprocessing
categorical_features = ['plan_combination']
numerical_features = features.columns.difference(categorical_features)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Streamlit UI
st.title("📞 Telecom Churn Prediction App")
st.write("Select a model and enter customer details to predict churn.")

# Dropdown for model selection
model_choice = st.selectbox("Choose a model:", ["Random Forest", "Logistic Regression", "Decision Tree"])

# Model setup
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, random_state=42)
else:
    model = DecisionTreeClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train model
pipeline.fit(features, target)

# Input form
st.subheader("Enter Customer Details")
user_input = {}
for col in numerical_features:
    user_input[col] = st.number_input(col, value=float(df[col].mean()), format="%.2f")

user_input['plan_combination'] = st.selectbox("Plan Combination", sorted(df['plan_combination'].unique()))

# Prediction
input_df = pd.DataFrame([user_input])
prediction = pipeline.predict(input_df)[0]
probability = pipeline.predict_proba(input_df)[0][1]

# Output
st.subheader("Prediction Result")
st.write(f"Churn Prediction: **{'YES' if prediction == 1 else 'NO'}**")
st.write(f"Churn Probability: **{probability:.2f}**")

