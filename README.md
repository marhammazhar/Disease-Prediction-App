# 🩺 Disease Prediction App

This project predicts diseases based on risk factors, symptoms, signs, and subtypes using machine learning models. It includes both a Jupyter notebook for analysis and a Streamlit app for interaction.

## 📂 Project Structure

- `streamlit_app.py`: Interactive web app using KNN to predict diseases.
- `Untitled1.ipynb`: Preprocessing, feature extraction, dimensionality reduction, and analysis.
- `disease_features.csv`: Original dataset with disease-related features.
- `encoded_output2.csv`: One-hot encoded version of the dataset.

## 🔍 Project Tasks

### ✅ Task 1: Feature Extraction
- Parse list-based columns (Symptoms, Signs, Risk Factors)
- Apply **TF-IDF vectorization** per column
- Compare with one-hot encoding

### ✅ Task 2: Dimensionality Reduction
- Applied **PCA** and **Truncated SVD**
- Visualized 2D plots of both encodings
- Compared cluster separability

### ✅ Task 3: Model Training
- Trained **KNN** with various distance metrics (Euclidean, Manhattan, Cosine)
- Evaluated using **5-fold cross-validation**
- Also trained **Logistic Regression** for comparison

### ✅ Task 4: Critical Analysis
- Compared TF-IDF vs. One-Hot
- Discussed clinical relevance and encoding limitations

## 🧠 Models Used

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

## 🚀 How to Run

### Run the Jupyter Notebook
```bash
Open Untitled1.ipynb and run all cells step-by-step.
