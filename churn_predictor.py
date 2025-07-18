import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os

def load_data(path):
    df = pd.read_csv(path)
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def preprocess(df):
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop customerID column if exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert total charges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Evaluation Complete")
    print(f"Accuracy Score: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return report

def save_report(report, filename="churn_report.txt"):
    with open(filename, "w") as f:
        f.write(report)
    print(f"📄 Report saved to {filename}")

def main():
    print("🚀 Starting Churn Prediction Pipeline...")

    path = "data/Telco-Customer-Churn.csv"
    if not os.path.exists(path):
        print(f"❌ Data file not found at: {path}")
        return

    df = load_data(path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    report = evaluate(model, X_test, y_test)
    save_report(report)

if __name__ == "__main__":
    main()
