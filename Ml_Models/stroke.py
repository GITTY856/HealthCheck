import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
from collections import Counter

print("\n--- Step 1: Data Cleaning ---")
data = pd.read_csv("stroke3.csv")
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
num_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
target = "stroke"

for c in cat_cols:
    data[c] = data[c].astype(str).str.strip()
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
print("First 5 cleaned lines:\n", data.head(), "\n")

print("\n--- Step 2: Data Preprocessing ---")
X = data[cat_cols + num_cols]
y = data[target].astype(int).values

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = enc.fit_transform(X[cat_cols])
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
X_processed = np.hstack([X_cat, X_num])
print("First 5 processed samples:\n", X_processed[:5], "\n")

print("\n--- Step 3: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

print("\n--- Step 4: Manual Random Forest Training ---")

def train_random_forest(X, y, n_trees=10, max_depth=None, sample_size=0.8, feature_ratio=0.8):
    trees = []
    m = X.shape[0]
    n_features = X.shape[1]
    for i in range(n_trees):
        idx = np.random.choice(m, int(m * sample_size), replace=True)
        feature_idx = np.random.choice(n_features, int(n_features * feature_ratio), replace=False)
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=i)
        tree.fit(X[idx][:, feature_idx], y[idx])
        trees.append((tree, feature_idx))
        print(f"Tree {i+1}/{n_trees} trained on {len(idx)} samples and {len(feature_idx)} features.")
    return trees

def predict_random_forest(trees, X):
    preds = []
    for tree, f_idx in trees:
        preds.append(tree.predict(X[:, f_idx]))
    preds = np.array(preds)
    final_pred = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
    return np.array(final_pred)

forest = train_random_forest(X_train, y_train, n_trees=20, max_depth=6)
y_pred = predict_random_forest(forest, X_test)
acc = np.mean(y_pred == y_test) * 100
print(f"\nTraining complete. Model Accuracy: {acc:.2f}%")

print("\n--- Step 5: Saving Model ---")
joblib.dump((forest, enc, scaler, cat_cols, num_cols), "manual_stroke_model.pkl")
print("Model saved as 'manual_stroke_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(inputs, result):
    print("\n--- Health Recommendation ---")
    if result == 1:
        print("⚠️ High risk of Stroke detected.")
        print("👨‍⚕️ Consult a Neurologist or Cardiologist immediately.")
        print("💡 Lifestyle & Health Tips:")
        print("- Control blood pressure and blood sugar levels.")
        print("- Quit smoking and avoid alcohol.")
        print("- Exercise regularly (30 min/day).")
        print("- Eat fruits, vegetables, and omega-3 rich foods.")
        print("- Monitor cholesterol and manage weight.")
    else:
        print("✅ Low risk of Stroke detected.")
        print("💪 Preventive Measures:")
        print("- Keep blood pressure and sugar under control.")
        print("- Stay physically active and avoid tobacco.")
        print("- Maintain balanced diet and reduce salt.")
        print("- Schedule routine checkups if above 40.")
    print("\n🧠 Stay heart-healthy to prevent strokes!\n")

def user_interface():
    print("\n--- Stroke Risk Prediction CLI ---")
    forest, enc, scaler, cat_cols, num_cols = joblib.load("manual_stroke_model.pkl")
    inputs = {}
    for col in cat_cols:
        val = input(f"{col}: ")
        inputs[col] = val
    for col in num_cols:
        val = float(input(f"{col}: "))
        inputs[col] = val
    user_df = pd.DataFrame([inputs])
    X_cat = enc.transform(user_df[cat_cols])
    X_num = scaler.transform(user_df[num_cols])
    X_user = np.hstack([X_cat, X_num])
    preds = []
    for tree, f_idx in forest:
        preds.append(tree.predict(X_user[:, f_idx]))
    final_pred = Counter([p[0] for p in preds]).most_common(1)[0][0]
    print(f"\nPredicted Outcome: {'Stroke Risk' if final_pred == 1 else 'No Stroke Risk'}")
    health_recommendations(inputs, final_pred)

user_interface()
