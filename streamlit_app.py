import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Title
st.title("🩺 Disease Classification using KNN (TF-IDF Based)")
st.write("This app lets you configure and evaluate a KNN model using TF-IDF features of diseases.")

# Sidebar settings
st.sidebar.header("🔧 Model Configuration")
k = st.sidebar.selectbox("Number of Neighbors (k)", [3, 5, 7])
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("disease_features.csv")
    df['Risk Factors'] = df['Risk Factors'].apply(eval).apply(lambda x: " ".join(x))
    df['Symptoms'] = df['Symptoms'].apply(eval).apply(lambda x: " ".join(x))
    df['Signs'] = df['Signs'].apply(eval).apply(lambda x: " ".join(x))
    df['combined'] = df['Risk Factors'] + " " + df['Symptoms'] + " " + df['Signs']
    return df

df = load_data()

# TF-IDF transformation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['combined'])

# Encode labels
y = df['Disease']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=k, metric=metric)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.subheader("📊 Classification Report")
report = classification_report(
    y_test,
    y_pred,
    labels=le.transform(le.classes_),  # Fixes the mismatch
    target_names=le.classes_,
    output_dict=False
)

st.text(report)

# Option to show raw data
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)
