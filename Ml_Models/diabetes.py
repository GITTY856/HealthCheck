import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("\n--- Step 1: Data Cleaning ---")
data = pd.read_csv("diabetes.csv")
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
data.fillna(data.median(), inplace=True)
print("First 5 lines of cleaned data:\n", data.head(), "\n")

print("\n--- Step 2: Data Preprocessing ---")
X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values.reshape(-1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("First 5 processed samples:\n", X[:5], "\n")

print("\n--- Step 3: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

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
            loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return w, b

weights, bias = train_logistic_regression(X_train, y_train)
y_pred_prob = 1 / (1 + np.exp(-(X_test.dot(weights) + bias)))
y_pred_label = (y_pred_prob >= 0.5).astype(int)
accuracy = np.mean(y_pred_label == y_test) * 100
print(f"\nTraining complete. Model Accuracy: {accuracy:.2f}%")

print("\n--- Step 5: Saving Model with joblib ---")
joblib.dump((weights, bias, scaler), "manual_logistic_model.pkl")
print("Model saved as 'manual_logistic_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(values, result, prob):
    print("\n--- Health Recommendation ---")
    glucose, bp, skin, insulin, bmi, dpf, age = values
    if result == 1:
        print("⚠️ You are likely to have Diabetes (Probability:", round(prob*100, 2), "%).")
        print("👨‍⚕️ Visit an Endocrinologist or Diabetologist.")
        print("💡 Maintain low sugar diet, regular exercise, and sleep well.")
    else:
        print("✅ You are not diabetic (Probability:", round(prob*100, 2), "%).")
        print("💪 Stay active, eat balanced meals, and get checkups regularly.")
    if glucose > 125: print("- High Glucose: Reduce sugar intake.")
    if bp > 130: print("- High BP: Limit salt and manage stress.")
    if bmi > 25: print("- High BMI: Exercise and control calorie intake.")
    if insulin > 200: print("- High Insulin: May indicate insulin resistance.")
    print("\n🩺 Keep monitoring your health regularly.\n")

def user_interface():
    print("\n--- Diabetes Prediction CLI ---")
    features = []
    feature_names = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    for name in feature_names:
        val = float(input(f"{name}: "))
        features.append(val)
    X_user = np.array(features).reshape(1, -1)
    weights, bias, scaler = joblib.load("manual_logistic_model.pkl")
    X_user_scaled = scaler.transform(X_user)
    y_pred_prob = 1 / (1 + np.exp(-(X_user_scaled.dot(weights) + bias)))[0, 0]
    y_pred_label = int(y_pred_prob >= 0.5)
    print(f"\nPrediction Probability: {y_pred_prob:.4f}")
    print("Predicted Outcome:", "Diabetic" if y_pred_label == 1 else "Non-Diabetic")
    health_recommendations(features, y_pred_label, y_pred_prob)

user_interface()
