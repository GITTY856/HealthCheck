import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("\n--- Step 1: Data Cleaning ---")
data = pd.read_csv("Pancreas.csv")
data.replace({'.F': np.nan, '.N': np.nan, 'M': np.nan, '9': np.nan}, inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
print("First 5 lines of cleaned data:\n", data.head(), "\n")

features = ['age','sex','cig_stat','pack_years','asp','ibup','diabetes_f',
            'gallblad_f','liver_comorbidity','bmi_curr','bmi_20','bmi_50',
            'panc_fh','fh_cancer']
target = 'panc_cancer'

X = data[features].copy()
y = data[target].astype(int).values  
print("\n--- Step 2: Data Preprocessing ---")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(X.median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("First 5 processed samples:\n", X_scaled[:5], "\n")

print("\n--- Step 3: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

print("\n--- Step 4: Logistic Regression (sklearn) Training ---")
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred) * 100
print(f"Training complete. Model Accuracy: {acc:.2f}%")

print("\n--- Step 5: Saving Model with joblib ---")
joblib.dump((model, scaler, features), "pancreas_model.pkl")
print("Model saved as 'pancreas_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(values, result, prob):
    print("\n--- Health Recommendation ---")
    if result == 1:
        print(f" High risk of Pancreatic Disease detected (Probability: {prob*100:.2f}%)")
        print(" Consult a Gastroenterologist or Hepatologist immediately.")
        print(" Suggestions:")
        print("- Avoid smoking and alcohol completely.")
        print("- Eat high-fiber foods, fruits, and whole grains.")
        print("- Avoid high-fat, processed, and fried foods.")
        print("- Maintain a healthy BMI and control blood sugar levels.")
    else:
        print(f" Low risk of Pancreatic Disease (Probability: {prob*100:.2f}%)")
        print(" Preventive Tips:")
        print("- Maintain healthy diet rich in vegetables and fruits.")
        print("- Avoid tobacco, alcohol, and excessive sugar.")
        print("- Stay active, maintain ideal body weight.")
        print("- Go for yearly health checkups if over 40 or with family history.")
    print("\n🩺 Stay proactive about your digestive health.\n")

def user_interface():
    print("\n--- Pancreatic Disease Prediction CLI ---")
    model, scaler, feature_names = joblib.load("pancreas_model.pkl")
    inputs = []
    for name in feature_names:
        val = float(input(f"{name}: "))
        inputs.append(val)
    X_user = np.array(inputs).reshape(1, -1)
    X_user_scaled = scaler.transform(X_user)
    prob = model.predict_proba(X_user_scaled)[0, 1]
    label = int(prob >= 0.5)
    print(f"\nPrediction Probability: {prob:.4f}")
    print("Predicted Outcome:", "Pancreatic Disease" if label == 1 else "No Disease Detected")
    health_recommendations(inputs, label, prob)

user_interface()
