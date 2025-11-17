#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter

import tkinter as tk
from tkinter import ttk, messagebox

# ============================================================
#  Support for manually trained DecisionTreeManual (COPD, maybe Liver)
# ============================================================

def entropy(y):
    classes = np.unique(y)
    ent = 0.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        ent -= p * np.log2(p + 1e-9)
    return ent

def information_gain(X_col, y, threshold):
    left = X_col < threshold
    right = X_col >= threshold
    if len(y[left]) == 0 or len(y[right]) == 0:
        return 0.0
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
        best_gain = 0.0
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


# ============================================================
#  Advice functions (ENHANCED)
# ============================================================

def copd_advice(pred_label, le):
    label = le.inverse_transform([pred_label])[0]
    sev = label.lower()

    lines = [f"Predicted COPD Severity: {label}"]

    # What this stage means + what to do
    if sev == "mild":
        lines.append("• Mild COPD — early stage lung damage, symptoms may be mild.")
        lines.append("• Focus on quitting smoking (if you smoke) and avoiding dust/pollution.")
        lines.append("• Start regular breathing exercises and light physical activity.")
    elif sev == "moderate":
        lines.append("• Moderate COPD — breathing may be difficult on exertion.")
        lines.append("• Use prescribed inhalers regularly and avoid respiratory infections.")
        lines.append("• Do pulmonary rehabilitation / breathing exercises consistently.")
    elif sev == "severe":
        lines.append("• Severe COPD — symptoms can limit daily activities.")
        lines.append("• Inhalers and medications must be taken regularly, even when feeling better.")
        lines.append("• Avoid all smoking, second-hand smoke, and indoor pollution.")
    elif sev == "very severe":
        lines.append("• Very Severe COPD — high risk of flare-ups and hospitalisation.")
        lines.append("• Oxygen therapy and continuous monitoring may be needed.")
        lines.append("• Avoid infections (mask, hand hygiene) and keep vaccinations (flu/pneumonia) updated.")
    else:
        lines.append("• COPD stage unclear — immediate detailed evaluation is advised.")

    # What to control most
    lines.append("")
    lines.append("Most important things to control:")
    lines.append("• Smoking / tobacco use (stop completely).")
    lines.append("• Exposure to dust, smoke, and pollution (use mask, improve ventilation).")
    lines.append("• Weight and physical activity (maintain healthy weight, walk daily if possible).")

    # Doctor recommendation
    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Pulmonologist (lung specialist).")
    lines.append("• If not available, consult a General Physician and ask for lung function tests (spirometry).")

    return "\n".join(lines)


def diabetes_advice(values, result, prob):
    glucose, bp, skin, insulin, bmi, dpf, age = values

    lines = []
    if result == 1:
        lines.append(f"You are likely to have Diabetes (Probability: {prob*100:.2f}%).")
        lines.append("• You should visit an Endocrinologist / Diabetologist soon.")
        lines.append("• Start monitoring fasting and post-meal sugar regularly.")
    else:
        lines.append(f"You are not diabetic according to this model (Probability: {(1-prob)*100:.2f}%).")
        lines.append("• Still maintain a healthy lifestyle and get routine checkups.")

    # Which parameters look more concerning
    alerts = []
    if glucose > 125:
        alerts.append("High Glucose (sugar) — control sugar intake, avoid sweets & sugary drinks.")
    if bp > 130:
        alerts.append("High Blood Pressure — reduce salt, manage stress, check BP regularly.")
    if bmi > 25:
        alerts.append("High BMI (overweight) — focus on weight loss by diet + exercise.")
    if insulin > 200:
        alerts.append("High Insulin — may indicate insulin resistance, needs medical advice.")
    if dpf > 0.7:
        alerts.append("Strong family history (DiabetesPedigreeFunction) — you are at higher lifetime risk.")

    lines.append("")
    if alerts:
        lines.append("Most important things you should control right now:")
        for a in alerts:
            lines.append(f"• {a}")
    else:
        lines.append("No major red flags in the inputs, but maintain a healthy routine.")

    # General lifestyle tips
    lines.append("")
    lines.append("Everyday steps you can start now:")
    lines.append("• Eat small, frequent, balanced meals (more vegetables, less refined sugar).")
    lines.append("• 30–40 minutes of walking or exercise on most days of the week.")
    lines.append("• Avoid smoking and limit alcohol.")
    lines.append("• Sleep 7–8 hours daily.")

    # Doctor recommendation
    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Endocrinologist / Diabetologist for long-term sugar control plan.")
    lines.append("• If not available, a General Physician can start basic diabetes management.")

    return "\n".join(lines)


def heart_advice(result):
    lines = []
    if result == 1:
        lines.append("High risk of Heart Disease detected.")
        lines.append("• You should consult a Cardiologist as soon as possible.")
        lines.append("• Do not ignore chest pain, shortness of breath, or pain radiating to arm/jaw.")
    else:
        lines.append("Low risk of Heart Disease detected by this model.")
        lines.append("• Still keep your heart healthy with lifestyle changes.")

    # What to control most
    lines.append("")
    lines.append("Most important things to control for heart health:")
    lines.append("• Blood pressure (target usually around 120/80 mmHg).")
    lines.append("• Cholesterol (especially LDL – bad cholesterol).")
    lines.append("• Blood sugar (if diabetic or pre-diabetic).")
    lines.append("• Smoking and alcohol (avoid smoking completely).")
    lines.append("• Weight and physical inactivity (aim for daily walking/exercise).")

    # Doctor recommendation
    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Cardiologist for detailed heart evaluation (ECG, Echo, TMT as advised).")
    lines.append("• If symptoms are severe or sudden (chest pain, sweating, breathlessness) → go to Emergency immediately.")

    return "\n".join(lines)


def kidney_advice(data, result):
    lines = []
    if result == 1:
        lines.append("Risk of Chronic Kidney Disease (CKD) detected.")
        lines.append("• Consult a Nephrologist (kidney specialist) as early as possible.")
        lines.append("• Kidney damage can worsen silently, so do not delay evaluation.")
    else:
        lines.append("No strong sign of Kidney Disease detected by this model.")
        lines.append("• Keep protecting your kidneys with healthy habits.")

    alerts = []

    # Key values and what to control
    bp = data.get("bp", None)
    sc = data.get("sc", None)
    bu = data.get("bu", None)
    hemo = data.get("hemo", None)
    sod = data.get("sod", None)
    pot = data.get("pot", None)

    if bp is not None and bp > 140:
        alerts.append("High Blood Pressure — major risk for kidney damage. Control BP strictly.")
    if sc is not None and sc > 1.4:
        alerts.append("High Creatinine — may indicate reduced kidney filtration.")
    if bu is not None and bu > 40:
        alerts.append("High Blood Urea — waste products building up in blood.")
    if hemo is not None and hemo < 12:
        alerts.append("Low Hemoglobin — possible anemia due to CKD.")
    if sod is not None and (sod < 135 or sod > 145):
        alerts.append("Abnormal Sodium — needs medical review.")
    if pot is not None and (pot < 3.5 or pot > 5.5):
        alerts.append("Abnormal Potassium — can affect heart rhythm, very important to correct.")

    lines.append("")
    if alerts:
        lines.append("Important things you should control based on your values:")
        for a in alerts:
            lines.append(f"• {a}")
    else:
        lines.append("No major lab red flags detected here, but follow doctor’s advice with regular tests.")

    # General kidney care tips
    lines.append("")
    lines.append("Daily kidney-friendly habits:")
    lines.append("• Keep blood pressure and diabetes under strict control.")
    lines.append("• Limit salt and very salty processed foods.")
    lines.append("• Avoid painkillers (NSAIDs) without doctor’s advice.")
    lines.append("• Stay hydrated, but do not overdrink if doctor has restricted fluids.")
    lines.append("• Avoid smoking and heavy alcohol use.")

    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Nephrologist for kidney function monitoring and treatment plan.")
    lines.append("• General Physician can guide primary BP/diabetes control until you see a specialist.")

    return "\n".join(lines)


def stroke_advice(result):
    lines = []
    if result == 1:
        lines.append("High risk of Stroke detected.")
        lines.append("• This is serious: you should consult a Neurologist or Cardiologist soon.")
        lines.append("• If you develop sudden weakness, slurred speech, facial drooping → go to Emergency immediately.")
    else:
        lines.append("Low risk of Stroke detected by this model.")
        lines.append("• Still keep stroke risk factors under good control.")

    lines.append("")
    lines.append("Most important things to control for stroke prevention:")
    lines.append("• Blood pressure (the single most important factor).")
    lines.append("• Diabetes and blood sugar.")
    lines.append("• Cholesterol levels.")
    lines.append("• Smoking (stop completely) and alcohol (avoid excess).")
    lines.append("• Weight, sedentary lifestyle, and stress.")

    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Neurologist for brain and stroke risk evaluation.")
    lines.append("• Cardiologist if you have heart rhythm issues or heart disease.")
    lines.append("• General Physician for routine BP/sugar checks and medications.")

    return "\n".join(lines)


def pancreas_advice(values, result, prob):
    # We don’t know all feature meanings here, so keep it more general
    lines = []
    if result == 1:
        lines.append(f"High risk of Pancreatic Disease (Probability: {prob*100:.2f}%).")
        lines.append("• Consult a Gastroenterologist / Hepatologist as soon as possible.")
        lines.append("• Do not ignore severe upper abdominal pain, vomiting, or weight loss.")
    else:
        lines.append(f"Low risk of Pancreatic Disease (Probability: {prob*100:.2f}%).")
        lines.append("• Continue to maintain a healthy digestive and metabolic lifestyle.")

    lines.append("")
    lines.append("Most important things you should control:")
    lines.append("• Alcohol intake (best to avoid completely).")
    lines.append("• Smoking (avoid fully).")
    lines.append("• High-fat and fried foods (can trigger pancreatic problems).")
    lines.append("• Obesity and uncontrolled diabetes (if present).")

    lines.append("")
    lines.append("Daily lifestyle suggestions:")
    lines.append("• Eat small, low-fat, frequent meals.")
    lines.append("• Maintain healthy body weight.")
    lines.append("• Stay hydrated and avoid heavy late-night meals.")

    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Gastroenterologist / Hepatologist for detailed evaluation and imaging (if required).")

    return "\n".join(lines)


def liver_advice(result, label_str=None):
    lines = []
    if label_str is not None:
        lines.append(f"Predicted Class: {label_str}")

    if result == 1:
        lines.append("Model suggests risk of Liver Disease.")
        lines.append("• Consult a Hepatologist or Gastroenterologist for further tests (LFT, Ultrasound, etc.).")
    else:
        lines.append("Low liver disease risk indicated by this model.")
        lines.append("• Keep protecting your liver with healthy habits.")

    lines.append("")
    lines.append("Most important things to control for liver health:")
    lines.append("• Alcohol intake (avoid or keep to minimum, as advised by doctor).")
    lines.append("• Body weight and fatty foods (fatty liver risk).")
    lines.append("• Diabetes and cholesterol levels.")
    lines.append("• Unnecessary medicines or herbal supplements that can harm the liver.")

    lines.append("")
    lines.append("Daily liver-friendly habits:")
    lines.append("• Eat more fruits, vegetables, and whole grains.")
    lines.append("• Avoid very oily, deep-fried, and junk food.")
    lines.append("• Do regular exercise to maintain healthy weight.")

    lines.append("")
    lines.append("Doctor to consult:")
    lines.append("• Hepatologist / Gastroenterologist.")
    lines.append("• General Physician can help arrange basic blood tests and imaging first.")

    return "\n".join(lines)


def rural_advice(disease, treatment, urgent):
    # Map disease to recommended doctor type
    doctor_map = {
        "Flu": "General Physician / Family Doctor",
        "Cold": "General Physician",
        "Bronchitis": "General Physician / Pulmonologist",
        "Dengue": "Physician or Infectious Disease Specialist",
        "Malaria": "Physician / Infectious Disease Specialist",
        "Typhoid": "Physician / Infectious Disease Specialist",
        "Food Poisoning": "General Physician / Gastroenterologist",
        "Heart Attack": "Cardiologist / Emergency Department",
        "Kidney Failure": "Nephrologist (Kidney Specialist)",
        "Stroke": "Neurologist / Emergency Department",
    }

    lines = [f"Possible Disease: {disease}",
             f"Suggested basic treatment/first steps: {treatment}"]

    if urgent:
        lines.append("⚠ This condition can be very serious. Please seek emergency care immediately if symptoms are severe.")
    else:
        lines.append("• Monitor symptoms; if they persist or worsen, visit a doctor soon.")

    # Which things to be careful about in general
    lines.append("")
    lines.append("General things you should watch and control:")
    lines.append("• Fever and dehydration (drink ORS / fluids if allowed).")
    lines.append("• Pain, breathing difficulty, chest pain, or confusion (these need urgent medical care).")
    lines.append("• Do not self-medicate heavily with antibiotics or painkillers without doctor advice.")

    lines.append("")
    doc = doctor_map.get(disease, "Nearest qualified doctor / Primary Health Centre")
    lines.append("Doctor to consult:")
    lines.append(f"• {doc}")

    return "\n".join(lines)



# ============================================================
#  Field hints (what to fill for each column)
# ============================================================

FIELD_HINTS = {
    # General
    "age": "Age in years (e.g. 45)",
    "Age": "Age in years (e.g. 45)",
    "AGE": "Age in years (e.g. 45)",
    "gender": "Male / Female",
    "Gender": "Male / Female",
    "Sex": "Male=1, Female=0 or as in dataset",
    "sex": "Male=1, Female=0 or as in dataset",

    # COPD typical
    "PackHistory": "Smoking history in pack-years (e.g. 10)",
    "MWT1": "6-minute walk test distance (meters)",
    "FEV1": "Forced Expiratory Volume in 1 second (L)",
    "FVC": "Forced Vital Capacity (L)",
    "CAT": "COPD Assessment Test score (0–40)",

    # Diabetes
    "Glucose": "Blood glucose (mg/dL), e.g. 110",
    "BloodPressure": "Blood pressure (mmHg), e.g. 120",
    "SkinThickness": "Triceps skinfold thickness (mm)",
    "Insulin": "Serum insulin (IU/mL)",
    "BMI": "Body Mass Index (kg/m²), e.g. 24.5",
    "DiabetesPedigreeFunction": "Diabetes family history score",
    "DiabetesPedigree": "Diabetes family history score",

    # Heart
    "RestingBP": "Resting BP (mmHg)",
    "Cholesterol": "Serum cholesterol (mg/dL)",
    "MaxHR": "Maximum heart rate achieved",
    "Oldpeak": "ST depression induced by exercise",
    "ChestPainType": "e.g. TA / ATA / NAP / ASY",
    "FastingBS": "Fasting blood sugar (0 or 1)",
    "RestingECG": "e.g. Normal / ST / LVH",
    "ExerciseAngina": "Y / N",
    "ST_Slope": "e.g. Up / Flat / Down",

    # Kidney (common column names)
    "bp": "Blood pressure (mmHg)",
    "sg": "Specific gravity (e.g. 1.020)",
    "al": "Albumin level (0–5)",
    "su": "Sugar level (0–5)",
    "bgr": "Blood glucose random (mg/dL)",
    "bu": "Blood urea (mg/dL)",
    "sc": "Serum creatinine (mg/dL)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (g/dL)",
    "pcv": "Packed cell volume (%)",
    "rc": "Red blood cell count (millions/cmm)",
    "wc": "White blood cell count (cells/cmm)",
    "htn": "Hypertension (yes/no or 1/0)",
    "dm": "Diabetes mellitus (yes/no or 1/0)",
    "cad": "Coronary artery disease (yes/no or 1/0)",
    "appet": "Appetite (good / poor)",
    "pe": "Pedal edema (yes/no)",
    "ane": "Anemia (yes/no)",

    # Liver (Indian liver dataset columns)
    "Total_Bilirubin": "Total bilirubin (mg/dL)",
    "Direct_Bilirubin": "Direct bilirubin (mg/dL)",
    "Alkaline_Phosphotase": "Alkaline phosphatase (IU/L)",
    "Alamine_Aminotransferase": "ALT (IU/L)",
    "Aspartate_Aminotransferase": "AST (IU/L)",
    "Total_Protiens": "Total proteins (g/dL)",
    "Albumin": "Albumin (g/dL)",
    "Albumin_and_Globulin_Ratio": "Albumin/Globulin ratio",

    # Stroke dataset
    "hypertension": "0 = No, 1 = Yes",
    "heart_disease": "0 = No, 1 = Yes",
    "ever_married": "Yes / No",
    "work_type": "Private / Self-employed / Govt_job / children / Never_worked",
    "Residence_type": "Urban / Rural",
    "avg_glucose_level": "Average glucose level (mg/dL)",
    "bmi": "Body Mass Index (kg/m²)",
    "smoking_status": "formerly smoked / never smoked / smokes / Unknown",

    # Pancreas (guessed)
    "bmi_curr": "Current BMI (kg/m²)",
    "diabetes_f": "Diabetes history (0/1 or as in dataset)",
    "smoking": "Smoking status",
    "smoking_status": "Smoking status",
}

def get_field_hint(col_name, is_numeric):
    """Return human-friendly hint text for a given column."""
    # try exact key
    if col_name in FIELD_HINTS:
        return FIELD_HINTS[col_name]
    # try case-insensitive match
    for k, v in FIELD_HINTS.items():
        if k.lower() == col_name.lower():
            return v
    # fallback generic
    if is_numeric:
        return "Numeric value (e.g. 0, 1.5, 120)"
    else:
        return "Category/text (e.g. Yes/No, Male/Female)"


# ============================================================
#  Scrollable Frame (full-height, centered content)
# ============================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#fdf6e3")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame")

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        max_width = min(800, event.width - 40)
        x = (event.width - max_width) // 2
        self.canvas.itemconfigure(self.window, width=max_width)
        self.canvas.coords(self.window, x, 0)


# ============================================================
#  Disease Form (with side hints)
# ============================================================

class DiseaseFormFrame(ttk.Frame):
    def __init__(self, parent, app, config):
        super().__init__(parent, style="App.TFrame")
        self.app = app
        self.config_data = config
        self.vars = {}
        self._build_ui()

    def _build_ui(self):
        disease_name = self.config_data["display_name"]
        dtype = self.config_data["type"]

        # Header
        header = ttk.Frame(self, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text=f"{disease_name} Prediction", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        back_btn = ttk.Button(header, text="← Back", style="Accent.TButton",
                              command=self.app.show_home)
        back_btn.grid(row=0, column=1, sticky="e")

        desc = ttk.Label(
            self,
            text="Fill patient details and click Predict. Hints are shown next to each field.",
            style="SubTitle.TLabel",
            wraplength=900,
            justify="left"
        )
        desc.grid(row=1, column=0, sticky="w", padx=20, pady=(5, 10))

        # Scrollable form (centered card)
        form_frame = ScrollableFrame(self)
        form_frame.grid(row=2, column=0, sticky="nsew", padx=0, pady=(0, 10))
        inner = form_frame.inner

        card = ttk.Frame(inner, style="Card.TFrame", padding=20)
        card.grid(row=0, column=0, pady=10, sticky="n")
        # col0: label, col1: entry, col2: hint text
        card.columnconfigure(0, weight=0)
        card.columnconfigure(1, weight=0)
        card.columnconfigure(2, weight=1)

        row = 0
        if dtype in ("COPD", "HEART", "KIDNEY", "STROKE", "LIVER"):
            cat_cols = self.config_data["cat_cols"]
            num_cols = self.config_data["num_cols"]

            if cat_cols:
                cat_label = ttk.Label(card, text="Categorical Inputs", style="Section.TLabel")
                cat_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
                row += 1
                for col in cat_cols:
                    lbl = ttk.Label(card, text=col, style="FieldLabel.TLabel")
                    lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

                    var = tk.StringVar()
                    entry = ttk.Entry(card, textvariable=var, width=22)
                    entry.grid(row=row, column=1, sticky="w", pady=3)

                    hint = get_field_hint(col, is_numeric=False)
                    hint_lbl = ttk.Label(card, text=hint, style="Hint.TLabel", wraplength=350, justify="left")
                    hint_lbl.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)

                    self.vars[col] = var
                    row += 1

            if num_cols:
                num_label = ttk.Label(card, text="Numeric Inputs", style="Section.TLabel")
                num_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 5))
                row += 1
                for col in num_cols:
                    lbl = ttk.Label(card, text=col, style="FieldLabel.TLabel")
                    lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

                    var = tk.StringVar()
                    entry = ttk.Entry(card, textvariable=var, width=22)
                    entry.grid(row=row, column=1, sticky="w", pady=3)

                    hint = get_field_hint(col, is_numeric=True)
                    hint_lbl = ttk.Label(card, text=hint, style="Hint.TLabel", wraplength=350, justify="left")
                    hint_lbl.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)

                    self.vars[col] = var
                    row += 1

        elif dtype in ("DIABETES", "PANCREAS"):
            feature_names = self.config_data["feature_names"]
            num_label = ttk.Label(card, text="Numeric Inputs", style="Section.TLabel")
            num_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
            row += 1
            for name in feature_names:
                lbl = ttk.Label(card, text=name, style="FieldLabel.TLabel")
                lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

                var = tk.StringVar()
                entry = ttk.Entry(card, textvariable=var, width=22)
                entry.grid(row=row, column=1, sticky="w", pady=3)

                hint = get_field_hint(name, is_numeric=True)
                hint_lbl = ttk.Label(card, text=hint, style="Hint.TLabel", wraplength=350, justify="left")
                hint_lbl.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)

                self.vars[name] = var
                row += 1

        elif dtype == "RURAL":
            symptom_names = self.config_data["symptom_names"]
            info_label = ttk.Label(card, text="Symptoms (1 = Yes, 0 = No)", style="Section.TLabel")
            info_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
            row += 1
            for name in symptom_names:
                lbl = ttk.Label(card, text=name, style="FieldLabel.TLabel")
                lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

                var = tk.StringVar(value="0")
                combo = ttk.Combobox(card, textvariable=var,
                                     values=["0", "1"], state="readonly", width=5)
                combo.grid(row=row, column=1, sticky="w", pady=3)

                hint_lbl = ttk.Label(card, text="0 = No, 1 = Yes", style="Hint.TLabel")
                hint_lbl.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)

                self.vars[name] = var
                row += 1

        # Predict button
        predict_frame = ttk.Frame(self, style="App.TFrame")
        predict_frame.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 5))
        predict_btn = ttk.Button(predict_frame, text="Predict", style="Primary.TButton",
                                 command=self.on_predict)
        predict_btn.grid(row=0, column=0, sticky="w")

        # Result & advice
        result_card = ttk.Frame(self, style="Card.TFrame", padding=15)
        result_card.grid(row=4, column=0, sticky="nsew", padx=20, pady=(0, 15))
        result_card.columnconfigure(0, weight=1)

        self.result_label = ttk.Label(result_card, text="", style="Result.TLabel")
        self.result_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.advice_text = tk.Text(
            result_card, height=8, wrap="word",
            bg="#fffaf0", fg="#111111", bd=1, relief="solid"
        )
        self.advice_text.grid(row=1, column=0, sticky="nsew")
        self.advice_text.configure(state="disabled")
        result_card.rowconfigure(1, weight=1)

        self.rowconfigure(2, weight=3)
        self.rowconfigure(4, weight=2)
        self.columnconfigure(0, weight=1)

    def _set_output(self, label_str, advice_str):
        self.result_label.config(text=f"Predicted: {label_str}")
        self.advice_text.configure(state="normal")
        self.advice_text.delete("1.0", tk.END)
        self.advice_text.insert(tk.END, advice_str)
        self.advice_text.configure(state="disabled")

    def on_predict(self):
        dtype = self.config_data["type"]
        try:
            if dtype == "COPD":
                self._predict_copd()
            elif dtype == "DIABETES":
                self._predict_diabetes()
            elif dtype == "HEART":
                self._predict_heart()
            elif dtype == "KIDNEY":
                self._predict_kidney()
            elif dtype == "STROKE":
                self._predict_stroke()
            elif dtype == "PANCREAS":
                self._predict_pancreas()
            elif dtype == "RURAL":
                self._predict_rural()
            elif dtype == "LIVER":
                self._predict_liver()
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ------------ Prediction methods (same logic as CLI) ------------

    def _predict_copd(self):
        model = self.config_data["model"]
        enc = self.config_data["enc"]
        scaler = self.config_data["scaler"]
        le = self.config_data["label_encoder"]
        cat_cols = self.config_data["cat_cols"]
        num_cols = self.config_data["num_cols"]

        data = {}
        for c in cat_cols:
            data[c] = self.vars[c].get().strip()
        for c in num_cols:
            val = self.vars[c].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {c}")
            data[c] = float(val)
        df = pd.DataFrame([data])
        X_cat = enc.transform(df[cat_cols]) if enc is not None and cat_cols else np.empty((1, 0))
        X_num = scaler.transform(df[num_cols])
        X_user = np.hstack([X_cat, X_num])
        pred = int(model.predict(X_user)[0])
        label_str = le.inverse_transform([pred])[0]
        advice = copd_advice(pred, le)
        self._set_output(label_str, advice)

    def _predict_diabetes(self):
        model = self.config_data["model"]
        scaler = self.config_data["scaler"]
        feature_names = self.config_data["feature_names"]
        values = []
        for name in feature_names:
            val = self.vars[name].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {name}")
            values.append(float(val))
        X = np.array(values).reshape(1, -1)
        Xs = scaler.transform(X)
        prob = model.predict_proba(Xs)[0, 1]
        label = int(prob >= 0.5)
        label_text = "Diabetic" if label == 1 else "Non-Diabetic"
        advice = diabetes_advice(values, label, prob)
        self._set_output(label_text, advice)

    def _predict_heart(self):
        forest = self.config_data["model"]
        enc = self.config_data["enc"]
        scaler = self.config_data["scaler"]
        cat_cols = self.config_data["cat_cols"]
        num_cols = self.config_data["num_cols"]

        ui = {}
        for c in cat_cols:
            ui[c] = self.vars[c].get().strip()
        for c in num_cols:
            val = self.vars[c].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {c}")
            ui[c] = float(val)
        df = pd.DataFrame([ui])
        X_cat = enc.transform(df[cat_cols]) if cat_cols else np.empty((1, 0))
        X_num = scaler.transform(df[num_cols])
        X_user = np.hstack([X_cat, X_num])
        preds = []
        for tree, f_idx in forest:
            preds.append(tree.predict(X_user[:, f_idx]))
        final = Counter([p[0] for p in preds]).most_common(1)[0][0]
        label_text = "Heart Disease Risk" if final == 1 else "No Heart Disease Risk"
        advice = heart_advice(final)
        self._set_output(label_text, advice)

    def _predict_kidney(self):
        forest = self.config_data["model"]
        enc = self.config_data["enc"]
        scaler = self.config_data["scaler"]
        cat_cols = self.config_data["cat_cols"]
        num_cols = self.config_data["num_cols"]

        data = {}
        for c in cat_cols:
            data[c] = self.vars[c].get().strip()
        for c in num_cols:
            val = self.vars[c].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {c}")
            data[c] = float(val)
        df = pd.DataFrame([data])
        X_cat = enc.transform(df[cat_cols]) if cat_cols else np.empty((1, 0))
        X_num = scaler.transform(df[num_cols])
        X_user = np.hstack([X_cat, X_num])
        preds = []
        for tree, f_idx in forest:
            preds.append(tree.predict(X_user[:, f_idx]))
        final = Counter([p[0] for p in preds]).most_common(1)[0][0]
        label_text = "CKD Risk" if final == 1 else "No CKD Risk"
        advice = kidney_advice(data, final)
        self._set_output(label_text, advice)

    def _predict_stroke(self):
        forest = self.config_data["model"]
        enc = self.config_data["enc"]
        scaler = self.config_data["scaler"]
        cat_cols = self.config_data["cat_cols"]
        num_cols = self.config_data["num_cols"]

        inputs = {}
        for c in cat_cols:
            inputs[c] = self.vars[c].get().strip()
        for c in num_cols:
            val = self.vars[c].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {c}")
            inputs[c] = float(val)
        df = pd.DataFrame([inputs])
        X_cat = enc.transform(df[cat_cols])
        X_num = scaler.transform(df[num_cols])
        X_user = np.hstack([X_cat, X_num])
        preds = []
        for tree, f_idx in forest:
            preds.append(tree.predict(X_user[:, f_idx]))
        final = Counter([p[0] for p in preds]).most_common(1)[0][0]
        label_text = "Stroke Risk" if final == 1 else "No Stroke Risk"
        advice = stroke_advice(final)
        self._set_output(label_text, advice)

    def _predict_pancreas(self):
        model = self.config_data["model"]
        scaler = self.config_data["scaler"]
        feature_names = self.config_data["feature_names"]

        vals = []
        for name in feature_names:
            val = self.vars[name].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {name}")
            vals.append(float(val))
        X = np.array(vals).reshape(1, -1)
        Xs = scaler.transform(X)
        prob = model.predict_proba(Xs)[0, 1]
        label = int(prob >= 0.5)
        text = "Pancreatic Disease" if label == 1 else "No Disease Detected"
        advice = pancreas_advice(vals, label, prob)
        self._set_output(text, advice)

    def _predict_rural(self):
        model = self.config_data["model"]
        symptom_names = self.config_data["symptom_names"]
        meds = self.config_data["medicines"]
        vals = []
        for name in symptom_names:
            v = self.vars[name].get().strip()
            if v not in ("0", "1"):
                raise ValueError(f"{name}: choose 0 or 1")
            vals.append(int(v))
        X = [vals]
        pred = model.predict(X)[0]
        treatment = meds.get(pred, "Consult a doctor")
        urgent = pred in ["Heart Attack", "Kidney Failure", "Stroke", "Malaria", "Typhoid", "Dengue"]
        advice = rural_advice(pred, treatment, urgent)
        self._set_output(pred, advice)

    def _predict_liver(self):
        model = self.config_data["model"]
        enc = self.config_data["enc"]
        scaler = self.config_data["scaler"]
        label_encoder = self.config_data.get("label_encoder")
        cat_cols = self.config_data["cat_cols"]
        num_cols = self.config_data["num_cols"]

        inputs = {}
        for c in cat_cols:
            inputs[c] = self.vars[c].get().strip()
        for c in num_cols:
            val = self.vars[c].get().strip()
            if val == "":
                raise ValueError(f"Enter numeric value for {c}")
            inputs[c] = float(val)
        df = pd.DataFrame([inputs])
        X_cat = enc.transform(df[cat_cols]) if cat_cols else np.empty((1, 0))
        X_num = scaler.transform(df[num_cols]) if num_cols else np.empty((1, 0))
        X_user = np.hstack([X_cat, X_num])
        pred = int(model.predict(X_user)[0])
        if label_encoder is not None:
            label_str = label_encoder.inverse_transform([pred])[0]
            label_text = label_str
        else:
            label_text = "Liver Disease Risk" if pred == 1 else "No Liver Disease Risk"
            label_str = None
        advice = liver_advice(pred, label_str)
        self._set_output(label_text, advice)


# ============================================================
#  Home Screen
# ============================================================

class HomeFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="App.TFrame")
        self.app = app
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left info
        left = ttk.Frame(self, style="App.TFrame", padding=(40, 40))
        left.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(
            left,
            text="Multi-Disease\nHealth Prediction",
            style="Hero.TLabel",
            justify="left"
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            left,
            text="Use your trained ML models to predict risk or severity\nfor multiple diseases.",
            style="SubTitle.TLabel",
            justify="left"
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(10, 20))

        # Right: selection card
        right_card = ttk.Frame(self, style="Card.TFrame", padding=20)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(0, 40), pady=40)
        right_card.columnconfigure(0, weight=1)
        right_card.columnconfigure(1, weight=1)

        header = ttk.Label(right_card, text="Select Disease", style="Section.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        buttons = [
            ("COPD", "COPD"),
            ("Diabetes", "DIABETES"),
            ("Heart Disease", "HEART"),
            ("Kidney Disease", "KIDNEY"),
            ("Liver Disease", "LIVER"),
            ("Stroke", "STROKE"),
            ("Pancreatic Disease", "PANCREAS"),
            ("Symptom-based (Rural)", "RURAL"),
        ]

        row = 1
        col = 0
        for text, key in buttons:
            b = ttk.Button(right_card, text=text, style="CardButton.TButton",
                           command=lambda k=key: self.app.open_disease(k))
            b.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            col += 1
            if col > 1:
                col = 0
                row += 1


# ============================================================
#  Main App
# ============================================================

class HealthGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Disease Health Prediction")
        self.configure(bg="#fdf6e3")

        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1100x700")

        style = ttk.Style(self)
        style.theme_use("clam")

        bg = "#fdf6e3"       # cream
        card_bg = "#fffaf0"  # lighter cream
        text = "#111111"
        subtext = "#555555"
        accent = "#2563eb"
        primary = "#16a34a"
        hint_color = "#777777"

        style.configure("App.TFrame", background=bg)
        style.configure("Card.TFrame", background=card_bg, relief="solid", borderwidth=1)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=text, background=bg)
        style.configure("Hero.TLabel", font=("Segoe UI", 26, "bold"), foreground=text, background=bg)
        style.configure("SubTitle.TLabel", font=("Segoe UI", 11), foreground=subtext, background=bg)
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent, background=card_bg)
        style.configure("FieldLabel.TLabel", font=("Segoe UI", 10), foreground=text, background=card_bg)
        style.configure("Result.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent, background=card_bg)
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground=hint_color, background=card_bg)

        style.configure("Primary.TButton",
                        font=("Segoe UI", 10, "bold"),
                        padding=6,
                        background=primary,
                        foreground="#ffffff")
        style.map("Primary.TButton",
                  background=[("active", "#15803d")])

        style.configure("Accent.TButton",
                        font=("Segoe UI", 9, "bold"),
                        padding=5,
                        background=accent,
                        foreground="#ffffff")
        style.map("Accent.TButton",
                  background=[("active", "#1d4ed8")])

        style.configure("CardButton.TButton",
                        font=("Segoe UI", 10),
                        padding=6,
                        background=card_bg,
                        foreground=text)
        style.map("CardButton.TButton",
                  background=[("active", "#f5e8c8")])

        self.container = ttk.Frame(self, style="App.TFrame")
        self.container.pack(fill="both", expand=True)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self.current_frame = None
        self.show_home()

    def switch_frame(self, frame):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = frame
        self.current_frame.grid(row=0, column=0, sticky="nsew")

    def show_home(self):
        frame = HomeFrame(self.container, self)
        self.switch_frame(frame)

    def open_disease(self, key):
        key = key.upper()
        cfg = self._load_config_for_disease(key)
        if cfg is None:
            return
        frame = DiseaseFormFrame(self.container, self, cfg)
        self.switch_frame(frame)

    # ---------------- Model loaders (same as CLI logic) ----------------

    def _load_config_for_disease(self, key):
        if key == "COPD":
            path = "manual_copd_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"COPD model file not found:\n{path}")
                return None
            tree, enc, scaler, le, cat_cols, num_cols = joblib.load(path)
            return {
                "type": "COPD",
                "display_name": "COPD",
                "model": tree,
                "enc": enc,
                "scaler": scaler,
                "label_encoder": le,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }

        if key == "DIABETES":
            candidates = ["diabetes.pkl", "logistic_model.pkl"]
            path = next((c for c in candidates if os.path.exists(c)), None)
            if path is None:
                messagebox.showerror("Model Not Found",
                                     "Diabetes model file not found (diabetes.pkl / logistic_model.pkl)")
                return None
            model, scaler = joblib.load(path)
            feature_names = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI',
                             'DiabetesPedigreeFunction','Age']
            return {
                "type": "DIABETES",
                "display_name": "Diabetes",
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
            }

        if key == "HEART":
            path = "manual_heart_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Heart model file not found:\n{path}")
                return None
            forest, enc, scaler, cat_cols, num_cols = joblib.load(path)
            return {
                "type": "HEART",
                "display_name": "Heart Disease",
                "model": forest,
                "enc": enc,
                "scaler": scaler,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }

        if key == "KIDNEY":
            path = "manual_kidney_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Kidney model file not found:\n{path}")
                return None
            forest, enc, scaler, cat_cols, num_cols = joblib.load(path)
            return {
                "type": "KIDNEY",
                "display_name": "Kidney Disease",
                "model": forest,
                "enc": enc,
                "scaler": scaler,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }

        if key == "STROKE":
            path = "manual_stroke_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Stroke model file not found:\n{path}")
                return None
            forest, enc, scaler, cat_cols, num_cols = joblib.load(path)
            return {
                "type": "STROKE",
                "display_name": "Stroke",
                "model": forest,
                "enc": enc,
                "scaler": scaler,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }

        if key == "PANCREAS":
            path = "manual_pancreas_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Pancreas model file not found:\n{path}")
                return None
            model, scaler, feature_names = joblib.load(path)
            return {
                "type": "PANCREAS",
                "display_name": "Pancreatic Disease",
                "model": model,
                "scaler": scaler,
                "feature_names": list(feature_names),
            }

        if key == "RURAL":
            path = "rural_healthcare.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Rural model file not found:\n{path}")
                return None
            model = joblib.load(path)
            symptom_names = [
                "Fever", "Cough", "Headache", "Fatigue", "Nausea",
                "Chest Pain", "Shortness of Breath",
                "Confusion/Dizziness", "Swelling in Legs", "Vision Problems"
            ]
            medicines = {
                "Flu": "Paracetamol, Cough Syrup, Warm fluids, Rest",
                "Malaria": "Antimalarial tablets, Hydration, Doctor consult",
                "Cold": "Steam inhalation, Ginger tea, Cough Syrup",
                "Typhoid": "Antibiotics, Hydration, Soft diet",
                "Heart Attack": "Aspirin, Immediate hospitalization",
                "Kidney Failure": "Low-salt diet, Avoid painkillers, Dialysis may be required",
                "Stroke": "Immediate hospitalization, Blood thinner medicines",
                "Dengue": "Paracetamol, Hydration, Platelet monitoring",
                "Bronchitis": "Cough Syrup, Inhaler, Warm fluids",
                "Food Poisoning": "ORS, Hydration, Light diet"
            }
            return {
                "type": "RURAL",
                "display_name": "Symptom-based (Rural Healthcare)",
                "model": model,
                "symptom_names": symptom_names,
                "medicines": medicines,
            }

        if key == "LIVER":
            path = "manual_liver_model.pkl"
            if not os.path.exists(path):
                messagebox.showerror("Model Not Found", f"Liver model file not found:\n{path}")
                return None
            try:
                data = joblib.load(path)
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load liver model:\n{e}")
                return None

            if isinstance(data, tuple):
                if len(data) == 5:
                    model, enc, scaler, cat_cols, num_cols = data
                    label_encoder = None
                elif len(data) == 6:
                    model, enc, scaler, label_encoder, cat_cols, num_cols = data
                else:
                    messagebox.showerror("Model Error",
                                         f"Unexpected liver model structure (tuple length {len(data)}).")
                    return None
            else:
                messagebox.showerror("Model Error", "Unexpected liver model type, expected a tuple.")
                return None

            cfg = {
                "type": "LIVER",
                "display_name": "Liver Disease",
                "model": model,
                "enc": enc,
                "scaler": scaler,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }
            if 'label_encoder' in locals():
                cfg["label_encoder"] = label_encoder
            return cfg

        messagebox.showerror("Error", f"Unknown disease key: {key}")
        return None


# ============================================================
#  Run
# ============================================================

if __name__ == "__main__":
    app = HealthGUI()
    app.mainloop()
