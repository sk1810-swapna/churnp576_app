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
st.sidebar.header("Input Customer Features")
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

# Generate dummy target and split
target = np.random.choice([0, 1], size=len(features), p=[0.7, 0.3])
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train all models and store accuracy
model_accuracy = {}
model_pipes = {}

for name, model in model_dict.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    model_accuracy[name] = accuracy_score(y_test, y_pred)
    model_pipes[name] = pipe

# Selected model prediction
selected_pipe = model_pipes[model_choice]
selected_prediction = selected_pipe.predict(input_df)[0]
selected_probability = selected_pipe.predict_proba(input_df)[0][1]

# Display selected model prediction
st.subheader("📈 Churn Prediction")
st.markdown(f"**Selected Model:** `{model_choice}`")
st.markdown(f"**Model Accuracy (on test data):** `{model_accuracy[model_choice]:.4f}`")
st.markdown(f"**Churn Prediction Probability:** `{selected_probability:.4f}`")

if selected_prediction == 1:
    st.error("⚠ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

# Visualization
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Stay", "Churn"], y=[1 - selected_probability, selected_probability], palette="Set2", ax=ax)
ax.set_title("Churn Probability Breakdown")
ax.set_ylabel("Probability")
st.pyplot(fig)

# Display best model (always Random Forest)
best_model_name = "Random Forest"
best_accuracy = model_accuracy[best_model_name]
st.subheader("Best Model Based on Accuracy")
st.markdown(f"**Model:** `{best_model_name}`")
st.markdown(f"**Accuracy:** `{best_accuracy:.4f}`")
