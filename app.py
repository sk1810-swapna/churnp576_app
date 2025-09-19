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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Page setup
st.set_page_config(page_title="📞 Churn Prediction App", layout="centered")
st.title("📞 Telecom Churn Prediction App")

# Sample dataset
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

# Sidebar inputs
st.sidebar.header("🔧 Input Customer Features")
model_choice = st.sidebar.selectbox("Choose Algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"])

user_input = {}
for col in numerical_features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

user_input['plan_combination'] = st.sidebar.selectbox("Plan Combination", sorted(df['plan_combination'].unique()))
input_df = pd.DataFrame([user_input])

# Append user input to dataset (simulate new customer)
df_augmented = pd.concat([features, input_df], ignore_index=True)
target_augmented = pd.concat([target, pd.Series([0])], ignore_index=True)  # Dummy label for input

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Model selection
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, random_state=42)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline and training
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

X_train, X_test, y_train, y_test = train_test_split(df_augmented, target_augmented, test_size=0.2, stratify=target_augmented, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Predict for user input
user_prediction = pipe.predict(input_df)[0]
user_probability = pipe.predict_proba(input_df)[0][1]

# Display prediction
st.subheader("📈 Churn Prediction")
st.markdown(f"**Selected Model:** `{model_choice}`")
st.markdown(f"**Churn Prediction Probability:** `{user_probability:.4f}`")

if user_prediction == 1:
    st.error("⚠️ This customer is likely to CHURN.")
else:
    st.success("✅ This customer is likely to STAY loyal.")

# Visualization
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Stay", "Churn"], y=[1 - user_probability, user_probability], palette="Set2", ax=ax)
ax.set_title("Churn Probability Breakdown")
ax.set_ylabel("Probability")
st.pyplot(fig)

# Display metrics
st.subheader("📊 Model Evaluation (with updated input)")
st.markdown(f"**Accuracy:** `{accuracy:.4f}`")
st.markdown(f"**Precision:** `{precision:.4f}`")
st.markdown(f"**Recall:** `{recall:.4f}`")
st.markdown(f"**F1-Score:** `{f1:.4f}`")
