import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
y = data[target].astype(int).values.reshape(-1, 1)

print("\n--- Step 2: Data Preprocessing ---")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(X.median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("First 5 processed samples:\n", X_scaled[:5], "\n")

print("\n--- Step 3: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

print("\n--- Step 4: Manual Logistic Regression Training ---")

def train_logistic_regression(X, y, learning_rate=0.05, epochs=2000):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    for epoch in range(epochs):
        z = X.dot(w) + b
        y_hat = 1 / (1 + np.exp(-z))
        error = y_hat - y
        w -= learning_rate * (X.T.dot(error) / m)
        b -= learning_rate * np.mean(error)
        if epoch % 200 == 0:
            loss = -np.mean(y*np.log(y_hat + 1e-9) + (1-y)*np.log(1-y_hat + 1e-9))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return w, b

weights, bias = train_logistic_regression(X_train, y_train)
y_pred_prob = 1 / (1 + np.exp(-(X_test.dot(weights) + bias)))
y_pred_label = (y_pred_prob >= 0.5).astype(int)
accuracy = np.mean(y_pred_label == y_test) * 100
print(f"\nTraining complete. Model Accuracy: {accuracy:.2f}%")

print("\n--- Step 5: Saving Model with joblib ---")
joblib.dump((weights, bias, scaler, features), "manual_pancreas_model.pkl")
print("Model saved as 'manual_pancreas_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(values, result, prob):
    print("\n--- Health Recommendation ---")
    if result == 1:
        print(f"⚠️ High risk of Pancreatic Disease detected (Probability: {prob*100:.2f}%)")
        print("👨‍⚕️ Consult a Gastroenterologist or Hepatologist immediately.")
        print("💡 Suggestions:")
        print("- Avoid smoking and alcohol completely.")
        print("- Eat high-fiber foods, fruits, and whole grains.")
        print("- Avoid high-fat, processed, and fried foods.")
        print("- Maintain a healthy BMI and control blood sugar levels.")
    else:
        print(f"✅ Low risk of Pancreatic Disease (Probability: {prob*100:.2f}%)")
        print("💪 Preventive Tips:")
        print("- Maintain healthy diet rich in vegetables and fruits.")
        print("- Avoid tobacco, alcohol, and excessive sugar.")
        print("- Stay active, maintain ideal body weight.")
        print("- Go for yearly health checkups if over 40 or with family history.")
    print("\n🩺 Stay proactive about your digestive health.\n")

def user_interface():
    print("\n--- Pancreatic Disease Prediction CLI ---")
    weights, bias, scaler, feature_names = joblib.load("manual_pancreas_model.pkl")
    inputs = []
    for name in feature_names:
        val = float(input(f"{name}: "))
        inputs.append(val)
    X_user = np.array(inputs).reshape(1, -1)
    X_user_scaled = scaler.transform(X_user)
    y_pred_prob = 1 / (1 + np.exp(-(X_user_scaled.dot(weights) + bias)))[0, 0]
    y_pred_label = int(y_pred_prob >= 0.5)
    print(f"\nPrediction Probability: {y_pred_prob:.4f}")
    print("Predicted Outcome:", "Pancreatic Disease" if y_pred_label == 1 else "No Disease Detected")
    health_recommendations(inputs, y_pred_label, y_pred_prob)

user_interface()
