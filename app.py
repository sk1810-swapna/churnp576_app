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

# Page setup
st.set_page_config(page_title="📞 Churn Prediction App", layout="centered")
st.title("📞 Telecom Churn Prediction App")

# Sample dataset (for UI and prediction only)
df = pd.DataFrame({
    "international_plan": np.random.choice([0, 1], size=100),
    "voice_mail_plan": np.random.choice([0, 1], size=100),
    "day_mins": np.random.uniform(100, 300, size=100),
    "day_calls": np.random.randint(50, 120, size=100),
    "evening_mins": np.random.uniform(100, 250, size=100),
    "evening_calls": np.random.randint(50, 100, size=100),
    "night_mins": np.random.uniform(80, 200, size=100),
    "night_calls": np.random.randint(40, 90, size=100),
    "international_mins": np.random.uniform(5, 20, size=100),
    "international_calls": np.random.randint(1, 10, size=100),
    "churn": np.random.choice([0, 1], size=100, p=[0.7, 0.3])
})

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

categorical_features = ['plan_combination']
numerical_features = features.columns.difference(categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Define models
model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Hardcoded notebook metrics
model_metrics = {
    "Logistic Regression": {
        "precision": 0.5975,
        "recall": 0.1967,
        "f1": 0.2960,
        "accuracy": 0.2960
    },
    "Decision Tree": {
        "precision": 0.7916,
        "recall": 0.8571,
        "f1": 0.8231,
        "accuracy": 0.8231
    },
    "Random Forest": {
        "precision": 0.9975,
        "recall": 0.8427,
        "f1": 0.9136,
        "accuracy": 0.9136
    }
}

# Train models for prediction
trained_pipelines = {}
for name, model in model_dict.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(features, target)
    trained_pipelines[name] = pipe

# Sidebar inputs
st.sidebar.header("🔧 Input Customer Features")
model_choice = st.sidebar.selectbox("Choose Algorithm", list(model_dict.keys()))

user_input = {}
for col in numerical_features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

user_input['plan_combination'] = st.sidebar.selectbox("Plan Combination", sorted(df['plan_combination'].unique()))
input_df = pd.DataFrame([user_input])

# Predict using selected model
selected_model = trained_pipelines[model_choice]
prediction = selected_model.predict(input_df)[0]
probability = selected_model.predict_proba(input_df)[0][1]

# Display prediction
st.subheader("📈 Churn Prediction")
st.markdown(f"**Selected Model:** `{model_choice}`")
st.markdown(f"**Churn Prediction Probability:** `{probability:.4f}`")

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

# Display metrics
metrics = model_metrics[model_choice]
st.subheader("📊 Evaluation Metrics (from notebook)")
st.markdown(f"**Accuracy:** `{metrics['accuracy']}`")
st.markdown(f"**Precision:** `{metrics['precision']}`")
st.markdown(f"**Recall:** `{metrics['recall']}`")
st.markdown(f"**F1-Score:** `{metrics['f1']}`")

# Best model by accuracy
best_model_name = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
best_accuracy = model_metrics[best_model_name]['accuracy']
st.subheader("🏆 Best Model Based on Accuracy")
st.markdown(f"**Model:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{best_accuracy}`")

