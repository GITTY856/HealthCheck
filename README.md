# HealthCheck
Ml model For Health prediction
# 🩺 HEALTHCHECK – AI Disease Prediction System

**HEALTHCHECK** is an AI-powered desktop application designed to predict the risk of multiple diseases using manually implemented machine learning algorithms and a modern Tkinter GUI interface.  
It provides early health insights, personalized recommendations, and helps users take preventive action.

---

## 🚀 Project Overview

This project integrates **seven disease prediction models** into one unified interface.  
Each model is built manually using core algorithms (Logistic Regression, Decision Tree, and Random Forest) without relying on prebuilt classifier functions — ensuring deeper understanding and full algorithmic control.

**Supported Diseases:**
- Diabetes  
- Liver Disease  
- Heart Disease  
- Kidney Disease  
- COPD (Chronic Obstructive Pulmonary Disease)  
- Pancreatic Disease  
- Stroke  

---

## 🧩 System Architecture

- **Frontend:** Tkinter & ttk (Python GUI)
- **Backend:** Manual ML Models (Logistic Regression, Decision Tree, Random Forest)
- **Data Handling:** NumPy, Pandas
- **Preprocessing:** OneHotEncoder, StandardScaler
- **Model Storage:** Joblib (.pkl files)
- **Execution:** Fully offline (no network dependency)

### 🏗️ Architecture Layers
1. **Data Layer:** Handles dataset cleaning, feature selection, and encoding.  
2. **Model Layer:** Contains trained models for each disease.  
3. **Application Layer:** GUI interface for user input, prediction, and results display.  

---

## 🧠 Machine Learning Models

| Disease | Algorithm | Model File | Accuracy (Approx.) |
|----------|------------|-------------|--------------------|
| Diabetes | Logistic Regression (Manual) | `manual_logistic_model.pkl` | 85% |
| Liver Disease | Decision Tree (Manual) | `manual_liver_model.pkl` | 80% |
| Heart Disease | Random Forest (Manual) | `manual_heart_model.pkl` | 88% |
| Kidney Disease | Random Forest (Manual) | `manual_kidney_model.pkl` | 86% |
| COPD | Decision Tree (Manual) | `manual_copd_model.pkl` | 78% |
| Pancreatic Disease | Logistic Regression (Manual) | `manual_pancreas_model.pkl` | 82% |
| Stroke | Random Forest (Manual) | `manual_stroke_model.pkl` | 87% |

---

## 👥 Team Contribution

| Member | Role | Tasks Completed |
|---------|------|-----------------|
| **Member 1 – Team Leader ** | Lead Developer | Developed COPD & Liver models (Decision Tree), designed complete GUI, integrated all models |
| **Member 2** | ML Developer | Built Heart & Stroke models (Random Forest), contributed to GUI enhancements |
| **Member 3** | Data Scientist | Created Diabetes & Pancreatic models (Logistic Regression), handled dataset cleaning |
| **Member 4** | Developer | Created Common Disease model (Decision Tree), assisted in testing and database setup |

---

## 🧪 Testing and Validation

- All models tested using **80:20 train-test split**.  
- Accuracy validated manually for each algorithm.  
- Random Forest models validated through multiple tree voting.  
- GUI tested for input, output, and model loading consistency.  
- Overall accuracy across models: **75–90%**.  

---

## 📊 Deliverables Progress

- ✅ Model Development (All Diseases) – *Completed*  
- ✅ Data Preprocessing and Cleaning – *Completed*  
- ✅ Model Integration in GUI – *Completed*  
- ⚙️ GUI Design Enhancement – *In Progress*  
- 🧠 Model Accuracy Optimization – *In Progress*  
- 🗄️ Database Connectivity & Storage – *Pending*  
- 📄 Documentation and Presentation – *Pending*  

---

## ⚙️ Installation & Usage

### **Requirements**
- Python 3.9+
- Libraries:  
  ```bash
  pip install numpy pandas scikit-learn joblib
