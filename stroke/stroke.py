# Brain Stroke Prediction - Complete Pipeline
# Dataset columns: id, gender, age, hypertension, heart_disease, ever_married, 
# work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# 1. DATA COLLECTION AND LOADING
# ============================================================
print("="*60)
print("STEP 1: DATA COLLECTION AND LOADING")
print("="*60)

# Load the dataset - Update filename if needed
df = pd.read_csv('stroke.csv')  # Change to your filename
print(f"✓ Dataset loaded successfully!")
print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

# ============================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("STEP 2: DATA CLEANING AND PREPROCESSING")
print("="*60)

# Basic dataset information
print("\nDataset Info:")
df.info()

print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("No missing values found!")

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"✓ Removed {duplicates} duplicate rows")

# Handle missing values in BMI
if df['bmi'].isnull().sum() > 0:
    median_bmi = df['bmi'].median()
    df['bmi'].fillna(median_bmi, inplace=True)
    print(f"✓ Filled {df['bmi'].isnull().sum()} missing BMI values with median: {median_bmi:.2f}")

# Remove 'id' column (not useful for prediction)
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("✓ Removed 'id' column")

# Display unique values for categorical columns
print("\nUnique values in categorical columns:")
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    print(f"\n{col}: {df[col].unique()}")
    print(f"Value counts:\n{df[col].value_counts()}")

# Handle 'Other' gender (if very few, can be removed or combined)
if 'gender' in df.columns and 'Other' in df['gender'].values:
    other_count = (df['gender'] == 'Other').sum()
    print(f"\n'Other' gender count: {other_count}")
    if other_count < 5:  # If very few, remove them
        df = df[df['gender'] != 'Other']
        print(f"✓ Removed {other_count} rows with 'Other' gender")

# Encode categorical variables
print("\n" + "-"*60)
print("Encoding Categorical Variables")
print("-"*60)

label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✓ Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n✓ All categorical variables encoded successfully!")
print("\nData after preprocessing:")
print(df.head(10))
print(f"\nFinal shape: {df.shape}")

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# Target variable distribution
print("\nTarget Variable (stroke) Distribution:")
stroke_counts = df['stroke'].value_counts()
print(stroke_counts)
stroke_percentage = df['stroke'].mean() * 100
print(f"\nStroke cases: {stroke_counts[1]} ({stroke_percentage:.2f}%)")
print(f"No stroke cases: {stroke_counts[0]} ({100-stroke_percentage:.2f}%)")
print(f"Class imbalance ratio: 1:{stroke_counts[0]/stroke_counts[1]:.2f}")

# Visualization 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
stroke_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Stroke Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Stroke (0=No, 1=Yes)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No Stroke', 'Stroke'], rotation=0)

# Pie chart
axes[1].pie(stroke_counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Stroke Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_stroke_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_stroke_distribution.png")
plt.close()

# Visualization 2: Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
            center=0, fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_correlation_matrix.png")
plt.close()

# Correlation with target variable
print("\nCorrelation with Target Variable (stroke):")
target_corr = correlation_matrix['stroke'].sort_values(ascending=False)
print(target_corr)

# Visualization 3: Feature distributions
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    df[df['stroke']==0][col].hist(bins=30, alpha=0.5, label='No Stroke', 
                                   ax=axes[idx], color='green')
    df[df['stroke']==1][col].hist(bins=30, alpha=0.5, label='Stroke', 
                                   ax=axes[idx], color='red')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('03_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_feature_distributions.png")
plt.close()

# Visualization 4: Box plots for outlier detection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(['age', 'avg_glucose_level', 'bmi']):
    df.boxplot(column=col, by='stroke', ax=axes[idx])
    axes[idx].set_title(f'{col} by Stroke Status')
    axes[idx].set_xlabel('Stroke')
    axes[idx].set_ylabel(col)
plt.suptitle('')
plt.tight_layout()
plt.savefig('04_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_boxplots.png")
plt.close()

print("\n✓ EDA completed! All visualizations saved.")

# ============================================================
# 4. SPLIT THE DATA
# ============================================================
print("\n" + "="*60)
print("STEP 4: SPLITTING THE DATA")
print("="*60)

# Separate features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")

# Split into train and test sets (80-20 split with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data split completed!")
print(f"Training set size: {X_train.shape[0]} samples ({(len(X_train)/len(X))*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({(len(X_test)/len(X))*100:.1f}%)")

print(f"\nTraining set stroke distribution:")
print(y_train.value_counts())
print(f"Stroke percentage in training: {y_train.mean()*100:.2f}%")

print(f"\nTest set stroke distribution:")
print(y_test.value_counts())
print(f"Stroke percentage in test: {y_test.mean()*100:.2f}%")

# Feature Scaling
print("\n" + "-"*60)
print("Feature Scaling")
print("-"*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ StandardScaler fitted on training data")
print("✓ Training and test data scaled")

# Handle class imbalance with SMOTE
print("\n" + "-"*60)
print("Handling Class Imbalance with SMOTE")
print("-"*60)

print(f"Before SMOTE - Training set distribution:")
print(f"Class 0 (No Stroke): {sum(y_train==0)}")
print(f"Class 1 (Stroke): {sum(y_train==1)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE - Training set distribution:")
print(f"Class 0 (No Stroke): {sum(y_train_balanced==0)}")
print(f"Class 1 (Stroke): {sum(y_train_balanced==1)}")
print(f"✓ Classes balanced successfully!")

# ============================================================
# 5. MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("STEP 5: MODEL TRAINING")
print("="*60)

models = {}

# Model 1: Logistic Regression
print("\n[1/3] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)
models['Logistic Regression'] = lr_model
print("✓ Logistic Regression trained successfully!")

# Model 2: Random Forest Classifier
print("\n[2/3] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
models['Random Forest'] = rf_model
print("✓ Random Forest trained successfully!")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('05_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_feature_importance.png")
plt.close()

# Model 3: Gradient Boosting Classifier
print("\n[3/3] Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train_balanced, y_train_balanced)
models['Gradient Boosting'] = gb_model
print("✓ Gradient Boosting trained successfully!")

print("\n" + "="*60)
print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)

# ============================================================
# 6. MODEL EVALUATION
# ============================================================
print("\n" + "="*60)
print("STEP 6: MODEL EVALUATION")
print("="*60)

results = {}
all_predictions = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    all_predictions[name] = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'06_confusion_matrix_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 06_confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

# ROC Curves for all models
print("\n" + "-"*60)
print("ROC Curves")
print("-"*60)

plt.figure(figsize=(10, 8))
for name, preds in all_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
    auc_score = roc_auc_score(y_test, preds['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('07_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_roc_curves.png")
plt.close()

# Model Comparison Summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print("\n" + results_df.to_string())

# Save comparison results
results_df.to_csv('model_comparison_results.csv')
print("\n✓ Saved: model_comparison_results.csv")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
results_df.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Models', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].set_xticklabels(results_df.index, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1])

# Heatmap
sns.heatmap(results_df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            ax=axes[1], cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
axes[1].set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Models', fontsize=12)
axes[1].set_ylabel('Metrics', fontsize=12)

plt.tight_layout()
plt.savefig('08_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_model_comparison.png")
plt.close()

# Find best model based on different metrics
print("\n" + "-"*60)
print("Best Models by Metric:")
print("-"*60)
for metric in results_df.columns:
    best_model = results_df[metric].idxmax()
    best_score = results_df[metric].max()
    print(f"  {metric}: {best_model} ({best_score:.4f})")

# Overall best model (based on F1-Score)
best_model_name = results_df['F1-Score'].idxmax()
best_model = models[best_model_name]
best_f1 = results_df.loc[best_model_name, 'F1-Score']

print(f"\n{'='*60}")
print(f"🏆 BEST OVERALL MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"{'='*60}")

# ============================================================
# 7. SAVE MODELS AND PREPROCESSORS
# ============================================================
print("\n" + "="*60)
print("STEP 7: SAVING MODELS AND PREPROCESSORS")
print("="*60)

# Save all models
for name, model in models.items():
    filename = f"model_{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"✓ Saved: {filename}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"\n✓ Saved best model: best_model.pkl ({best_model_name})")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Saved: label_encoders.pkl")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Saved: feature_names.pkl")

# Save SMOTE object
joblib.dump(smote, 'smote.pkl')
print("✓ Saved: smote.pkl")

# Save results dataframe
results_df.to_pickle('results_dataframe.pkl')
print("✓ Saved: results_dataframe.pkl")

# Create a summary report
summary_report = f"""
{'='*60}
BRAIN STROKE PREDICTION - FINAL REPORT
{'='*60}

Dataset Information:
- Total samples: {len(df)}
- Features: {len(X.columns)}
- Training samples: {len(X_train)}
- Testing samples: {len(X_test)}
- Stroke cases: {sum(y==1)} ({(sum(y==1)/len(y))*100:.2f}%)

Best Model: {best_model_name}
- F1-Score: {best_f1:.4f}
- Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f}
- Precision: {results_df.loc[best_model_name, 'Precision']:.4f}
- Recall: {results_df.loc[best_model_name, 'Recall']:.4f}
- ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}

Top 5 Important Features:
{feature_importance.head().to_string(index=False)}

Files Saved:
- Models: model_*.pkl, best_model.pkl
- Preprocessors: scaler.pkl, label_encoders.pkl, smote.pkl
- Results: model_comparison_results.csv, results_dataframe.pkl
- Visualizations: 8 PNG files

{'='*60}
"""

with open('project_summary.txt', 'w') as f:
    f.write(summary_report)

print("\n✓ Saved: project_summary.txt")
print(summary_report)

print("\n" + "="*60)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 All files saved in current directory:")
print("   - 3 trained models (.pkl)")
print("   - 1 best model (.pkl)")
print("   - 5 preprocessor files (.pkl)")
print("   - 8 visualization plots (.png)")
print("   - 2 result files (.csv, .txt)")
print("\n💡 To use the model for predictions:")
print("   model = joblib.load('best_model.pkl')")
print("   scaler = joblib.load('scaler.pkl')")
print("   # Then: predictions = model.predict(scaler.transform(new_data))")
print("\n" + "="*60)