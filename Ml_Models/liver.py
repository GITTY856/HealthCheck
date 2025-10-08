import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

print("\n--- Step 1: Data Cleaning ---")
data = pd.read_csv("indian_liver_patient.csv")
data = data.rename(columns={"Dataset": "liver_disease"})
data["liver_disease"] = data["liver_disease"].map({1: 1, 2: 0})
data["Albumin_and_Globulin_Ratio"] = pd.to_numeric(data["Albumin_and_Globulin_Ratio"], errors="coerce")
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)
print("First 5 lines of cleaned data:\n", data.head(), "\n")

target = "liver_disease"
X = data.drop(columns=[target])
y = data[target].astype(int).values

cat_cols = ["Gender"]
num_cols = [c for c in X.columns if c not in cat_cols]

print("\n--- Step 2: Data Preprocessing ---")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = enc.fit_transform(X[cat_cols])
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
    left_idx = X_col < threshold
    right_idx = X_col >= threshold
    if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
        return 0
    p = len(y[left_idx]) / len(y)
    gain = entropy(y) - (p * entropy(y[left_idx]) + (1 - p) * entropy(y[right_idx]))
    return gain

class DecisionTreeManual:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.threshold = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y):
        if len(np.unique(y)) == 1 or self.depth >= self.max_depth:
            self.label = np.round(np.mean(y))
            return
        best_gain = 0
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                gain = information_gain(X[:, feature], y, t)
                if gain > best_gain:
                    best_gain = gain
                    self.feature_index = feature
                    self.threshold = t
        if best_gain == 0:
            self.label = np.round(np.mean(y))
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
joblib.dump((tree, enc, scaler, cat_cols, num_cols), "manual_liver_model.pkl")
print("Model saved as 'manual_liver_model.pkl'")

print("\n--- Step 6: CLI-Based User Prediction & Health Advice ---")

def health_recommendations(data, result):
    print("\n--- Health Recommendation ---")
    if result == 1:
        print("⚠️ Patient is at risk of Liver Disease.")
        print("👨‍⚕️ Consult a Hepatologist or Gastroenterologist.")
        print("💡 Lifestyle & Health Tips:")
        print("- Avoid alcohol and fatty foods.")
        print("- Eat a high-fiber diet with fresh fruits and vegetables.")
        print("- Exercise regularly to maintain body weight.")
        print("- Control diabetes and cholesterol levels.")
        print("- Avoid self-medication or herbal supplements.")
        if data["Total_Bilirubin"] > 1.2:
            print("- High Total Bilirubin: Possible liver dysfunction.")
        if data["Direct_Bilirubin"] > 0.3:
            print("- High Direct Bilirubin: Possible bile blockage.")
        if data["Alkaline_Phosphotase"] > 120:
            print("- High ALP: Possible liver or bone disorder.")
        if data["Alamine_Aminotransferase"] > 40 or data["Aspartate_Aminotransferase"] > 40:
            print("- Elevated SGOT/SGPT: Possible liver cell injury.")
    else:
        print("✅ No sign of Liver Disease detected.")
        print("💪 Preventive Tips:")
        print("- Avoid alcohol and smoking.")
        print("- Drink plenty of water and eat clean foods.")
        print("- Maintain healthy BMI and avoid junk food.")
        print("- Get regular liver checkups if you take medications frequently.")
    print("\n🩺 Take care of your liver—it filters and detoxifies everything!\n")

def user_interface():
    print("\n--- Liver Disease Prediction CLI ---")
    tree, enc, scaler, cat_cols, num_cols = joblib.load("manual_liver_model.pkl")
    data = {}
    for c in cat_cols:
        val = input(f"{c}: ")
        data[c] = val
    for c in num_cols:
        val = float(input(f"{c}: "))
        data[c] = val
    df = pd.DataFrame([data])
    X_cat = enc.transform(df[cat_cols])
    X_num = scaler.transform(df[num_cols])
    X_user = np.hstack([X_cat, X_num])
    pred = int(tree.predict(X_user)[0])
    print(f"\nPredicted Outcome: {'Liver Disease Risk' if pred == 1 else 'No Liver Disease Risk'}")
    health_recommendations(data, pred)

user_interface()
