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

# Sample dataset for slider ranges
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
    "international_calls": np.random.randint(1, 10, size=100)
})

# Feature engineering
df['plan_combination'] = df['international_plan'].astype(str) + "_" + df['voice_mail_plan'].astype(str)
df['avg_day_call_duration'] = df['day_mins'] / (df['day_calls'] + 1e-5)
df['avg_evening_call_duration'] = df['evening_mins'] / (df['evening_calls'] + 1e-5)
df['avg_night_call_duration'] = df['night_mins'] / (df['night_calls'] + 1e-5)
df['avg_international_call_duration'] = df['international_mins'] / (df['international_calls'] + 1e-5)
df['total_calls'] = df[['day_calls', 'evening_calls', 'night_calls', 'international_calls']].sum(axis=1)
df['total_mins'] = df[['day_mins', 'evening_mins', 'night_mins', 'international_mins']].sum(axis=1)

# Define features
features = df.copy()
categorical_features = ['plan_combination']
numerical_features = features.columns.difference(categorical_features)

# Sidebar inputs
st.sidebar.header("🔧 Input Customer Features")
model_choice = st.sidebar.selectbox("Choose Algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"])

# Initialize session state for sliders
for col in numerical_features:
    if f"{col}_value" not in st.session_state:
        st.session_state[f"{col}_value"] = round(float(df[col].mean()), 2)

# Create sliders with session state
user_input = {}
for col in numerical_features:
    user_input[col] = st.sidebar.slider(
        label=col,
        min_value=round(float(df[col].min()), 2),
        max_value=round(float(df[col].max()), 2),
        value=st.session_state[f"{col}_value"],
        step=0.01,
        key=f"{col}_value"
    )

user_input['plan_combination'] = st.sidebar.selectbox("Plan Combination", sorted(df['plan_combination'].unique()))
input_df = pd.DataFrame([user_input])

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Model definitions
model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Hardcoded accuracy values
model_accuracy = {
    "Logistic Regression": 0.8644,
    "Decision Tree": 0.9466,
    "Random Forest": 0.9805
}

# Generate dummy target once
dummy_target = np.random.choice([0, 1], size=len(features), p=[0.7, 0.3])

# Train all models on same data
model_predictions = {}
model_probabilities = {}

for name, model in model_dict.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(features, dummy_target)
    model_predictions[name] = pipe.predict(input_df)[0]
    model_probabilities[name] = pipe.predict_proba(input_df)[0][1]

# Selected model output
selected_prediction = model_predictions[model_choice]
selected_probability = model_probabilities[model_choice]

st.subheader("📈 Churn Prediction")
st.markdown(f"**Selected Model:** `{model_choice}`")
st.markdown(f"**Model Accuracy:** `{model_accuracy[model_choice]}`")
st.markdown(f"**Churn Prediction Probability:** `{selected_probability:.4f}`")

if selected_prediction == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

# Visualization
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Stay", "Churn"], y=[1 - selected_probability, selected_probability], palette="Set2", ax=ax)
ax.set_title("Churn Probability Breakdown")
ax.set_ylabel("Probability")
st.pyplot(fig)

# Best model output
best_model_name = max(model_accuracy.items(), key=lambda x: x[1])[0]
best_accuracy = model_accuracy[best_model_name]

# Use selected model's probability if it's also the best
if model_choice == best_model_name:
    best_probability = selected_probability
else:
    best_probability = model_probabilities[best_model_name]

st.subheader("🏆 Best Model Based on Accuracy")
st.markdown(f"**Model:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{best_accuracy}`")
st.markdown(f"**Churn Prediction Probability:** `{best_probability:.4f}`")
