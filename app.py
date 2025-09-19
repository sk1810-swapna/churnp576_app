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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page setup
st.set_page_config(page_title="📞 Churn Prediction App", layout="centered")
st.title("📞 Telecom Churn Prediction App")

# ✅ Embedded sample dataset
df = pd.DataFrame({
    "international_plan": [0, 1, 0, 1],
    "voice_mail_plan": [1, 0, 1, 0],
    "day_mins": [300, 120, 250, 180],
    "day_calls": [100, 80, 90, 85],
    "evening_mins": [200, 150, 180, 160],
    "evening_calls": [90, 85, 88, 82],
    "night_mins": [150, 130, 140, 135],
    "night_calls": [80, 75, 78, 76],
    "international_mins": [10, 20, 15, 12],
    "international_calls": [3, 5, 4, 3],
    "churn": [0, 1, 0, 1]
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

# Train models and compute metrics
model_metrics = {}
trained_pipelines = {}

for name, model in model_dict.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    model_metrics[name] = metrics
    trained_pipelines[name] = pipe

# Display full model comparison table
st.subheader("📊 Model Performance Comparison")
metrics_df = pd.DataFrame(model_metrics).T
metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
st.table(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

# Select best model by F1-score
best_model_name = max(model_metrics.items(), key=lambda x: x[1]["F1-Score"])[0]
best_model = trained_pipelines[best_model_name]
best_metrics = model_metrics[best_model_name]

# Sidebar inputs
st.sidebar.header("🔧 Input Customer Features")
model_choice = st.sidebar.selectbox("Choose Algorithm for Prediction", list(model_dict.keys()))

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
selected_metrics = model_metrics[model_choice]

# ✅ Display churn prediction
st.subheader("📈 Churn Prediction")
st.markdown(f"**Selected Model:** `{model_choice}`")
st.markdown(f"**Churn Prediction Probability:** `{probability * 100:.2f}%`")

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

# ✅ Display selected model metrics
st.subheader("📌 Selected Model Performance")
st.markdown(f"**Accuracy:** `{selected_metrics['Accuracy']:.4f}`")
st.markdown(f"**Precision:** `{selected_metrics['Precision']:.4f}`")
st.markdown(f"**Recall:** `{selected_metrics['Recall']:.4f}`")
st.markdown(f"**F1-Score:** `{selected_metrics['F1-Score']:.4f}`")

# ✅ Display best model summary
st.subheader("🏆 Best Model Based on F1-Score")
st.markdown(f"**Model:** `{best_model_name}`")
st.markdown(f"**F1-Score:** `{best_metrics['F1-Score']:.4f}`")
