"""
Handwritten Character Recognition using Machine Learning (MNIST)
Task 2 - Standalone Training & Inference Script
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

def load_data():
    """Phase 1 & Phase 2: Data Collection and Preprocessing"""
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path = os.path.join(DATASET_DIR, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_raw = train_df.iloc[:, 1:].values / 255.0
    y_train_raw = train_df.iloc[:, 0].values
    
    X_test_raw = test_df.iloc[:, 1:].values / 255.0
    y_test_raw = test_df.iloc[:, 0].values
    
    return X_train_raw, y_train_raw, X_test_raw, y_test_raw

def train_model(X_train, y_train):
    """Phase 4 & Phase 5: Model Building and Training"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=25,
        activation='relu',
        solver='adam',
        random_state=42,
        verbose=True
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Phase 6 & Phase 7: Model Testing and Performance Evaluation"""
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='macro')
    rec = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    
    print("\n" + "="*50)
    print("      PERFORMANCE EVALUATION METRICS      ")
    print("="*50)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions

def save_model(model):
    """Save trained model to disk"""
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to: {MODEL_PATH}")

def main():
    print("Starting Handwritten Character Recognition Workflow...")
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
