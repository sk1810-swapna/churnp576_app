import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page setup
st.set_page_config(page_title="📞 Churn Prediction App", layout="centered")
st.title("📞 Telecom Churn Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your churn dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Feature engineering
    df['plan_combination'] = df['international_plan'].astype(str) + "_" + df['voice_mail_plan'].astype(str)
    df['avg_day_call_duration'] = df['day_mins'] / (df['day_calls'] + 1e-5)
    df['avg_evening_call_duration'] = df['evening_mins'] / (df['evening_calls'] + 1e-5)
    df['avg_night_call_duration'] = df['night_mins'] / (df['night_calls'] + 1e-5)
    df['avg_international_call_duration'] = df['international_mins'] / (df['international_calls'] + 1e-5)
    df['total_calls'] = df[['day_calls', 'evening_calls', 'night_calls', 'international_calls']].sum(axis=1)
    df['total_mins'] = df[['day_mins', 'evening_mins', 'night_mins', 'international_mins']].sum(axis=1)

    # Define features and target
    target = df['churn']
    features = df.drop(['churn'], axis=1)

    # Column types
    categorical_features = ['plan_combination']
    numerical_features = features.columns.difference(categorical_features)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define models
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train all models and compute accuracy
    model_scores = {}
    trained_pipelines = {}

    for name, model in model_dict.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        model_scores[name] = acc
        trained_pipelines[name] = pipe

    # Sort models by accuracy (prefer Random Forest in tie)
    sorted_models = sorted(model_scores.items(), key=lambda x: (-x[1], x[0] != "Random Forest"))
    best_model_name = sorted_models[0][0]
    best_model = trained_pipelines[best_model_name]
    best_accuracy = model_scores[best_model_name]

    # Sidebar input
    st.sidebar.header("🔧 Input Customer Features")
    user_input = {}
    for col in numerical_features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

    user_input['plan_combination'] = st.sidebar.selectbox("Plan Combination", sorted(df['plan_combination'].unique()))
    input_df = pd.DataFrame([user_input])

    # Predict with best model
    prediction = best_model.predict(input_df)[0]
    probability = best_model.predict_proba(input_df)[0][1]

    # Display results
    st.subheader("📊 Prediction Result")
    st.markdown(f"**Best Model Selected:** `{best_model_name}`")
    st.markdown(f"**Model Accuracy:** `{best_accuracy:.4f}`")

    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is likely to STAY loyal.")

    # Visualization
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=["Stay", "Churn"], y=[1 - probability, probability], palette="Set2", ax=ax)
    ax.set_title("Churn Probability Breakdown")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file to begin.")
