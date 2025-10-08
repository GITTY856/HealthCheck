import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import joblib

print("\n--- Step 1: Data Cleaning ---")
data = pd.read_csv("Copd_1500.csv")

num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
for c in cat_cols:
    data[c] = data[c].astype(str).str.strip().str.title()

target = "COPDSEVERITY" if "COPDSEVERITY" in data.columns else data.columns[-1]
data.dropna(subset=[target], inplace=True)

data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
data = data.drop_duplicates()

le = LabelEncoder()
data[target] = le.fit_transform(data[target])
print("First 5 lines of cleaned data:\n", data.head(), "\n")

X = data.drop(columns=[target])
y = data[target].values

# Correct feature identification
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

print("\nDetected categorical columns:", cat_cols)
print("Detected numeric columns:", num_cols)

print("\n--- Step 2: Data Preprocessing ---")
if len(cat_cols) > 0:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(X[cat_cols])
else:
    enc = None
    X_cat = np.empty((len(X), 0))
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
X_processed = np.hstack([X_cat, X_num])
print("First 5 processed samples:\n", X_processed[:5], "\n")

print("\n--- Step 3: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

print("\n--- Step 4: Manual Decision Tree Training ---")

def entropy(y):
    classes = np.unique(y)
    ent = 0
    for c in classes:
        p = np.sum(y == c) / len(y)
        ent -= p * np.log2(p + 1e-9)
    return ent

def information_gain(X_col, y, threshold):
    left = X_col < threshold
    right = X_col >= threshold
    if len(y[left]) == 0 or len(y[right]) == 0:
        return 0
    p = len(y[left]) / len(y)
    gain = entropy(y) - (p * entropy(y[left]) + (1 - p) * entropy(y[right]))
    return gain

class DecisionTreeManual:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y):
        if len(np.unique(y)) == 1 or self.depth >= self.max_depth:
            self.label = np.argmax(np.bincount(y))
            return
        best_gain = 0
        n_features = X.shape[1]
        for i in range(n_features):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                gain = information_gain(X[:, i], y, t)
                if gain > best_gain:
                    best_gain = gain
                    self.feature_index = i
                    self.threshold = t
        if best_gain == 0:
            self.label = np.argmax(np.bincount(y))
            return
        left_idx = X[:, self.feature_index] < self.threshold
        right_idx = X[:, self.feature_index] >= self.threshold
        self.left = DecisionTreeManual(self.depth + 1, self.max_depth)
        self.left.fit(X[left_idx], y[left_idx])
        self.right = DecisionTreeManual(self.depth + 1, self.max_depth)
        self.right.fit(X[right_idx], y[right_idx])

    def predict_one(self, x):
        if self.label is not None:
            return self.label
        if x[self.feature_index] < self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

tree = DecisionTreeManual(max_depth=6)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
acc = np.mean(y_pred == y_test) * 100
print(f"\nTraining complete. Model Accuracy: {acc:.2f}%")

print("\n--- Step 5: Saving Model ---")
joblib.dump((tree, enc, scaler, le, cat_cols, num_cols), "manual_copd_model.pkl")
print("Model saved as 'manual_copd_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(pred_label):
    label = le.inverse_transform([pred_label])[0]
    print(f"\nPredicted COPD Severity: {label}")
    if label.lower() == "mild":
        print("💨 Mild COPD — early stage, avoid smoking, maintain clean air, and do breathing exercises.")
    elif label.lower() == "moderate":
        print("🌬 Moderate COPD — maintain medications and pulmonary exercises regularly.")
    elif label.lower() == "severe":
        print("⚠️ Severe COPD — inhaler therapy and regular pulmonologist visits required.")
    elif label.lower() == "very severe":
        print("🚨 Very Severe COPD — may require oxygen therapy and constant monitoring.")
    else:
        print("COPD stage unclear — consult a Pulmonologist immediately.")
    print("\n👨‍⚕️ Recommended Doctor: Pulmonologist (Lung Specialist)\n")

def user_interface():
    print("\n--- COPD Severity Prediction CLI ---")
    tree, enc, scaler, le, cat_cols, num_cols = joblib.load("manual_copd_model.pkl")
    data = {}
    for c in cat_cols:
        val = input(f"{c}: ")
        data[c] = val
    for c in num_cols:
        val = float(input(f"{c}: "))
        data[c] = val
    df = pd.DataFrame([data])
    if enc is not None and len(cat_cols) > 0:
        X_cat = enc.transform(df[cat_cols])
    else:
        X_cat = np.empty((1, 0))
    X_num = scaler.transform(df[num_cols])
    X_user = np.hstack([X_cat, X_num])
    pred = int(tree.predict(X_user)[0])
    health_recommendations(pred)

user_interface()
