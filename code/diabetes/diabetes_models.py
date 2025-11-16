import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC                     
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added confusion_matrix
from joblib import dump

# --- 1. Load Data and Define X/y ---
try:
    df = pd.read_csv('NHANES.csv')
except FileNotFoundError:
    print("Error: 'NHANES_diabetes_Completed.csv' not found. Make sure the file is in the same directory.")
    exit()

# FIX: REMOVE ALL GLUCOSE AND HBA1C-BASED FEATURES 
features_to_drop = [
   'DIABETES', 
   'LBXGLU',                     
   'LBXGLU_SQ', 'LBXGH_SQ',              
   'INSULIN_GLUCOSE_RATIO'               
]

X = df.drop(features_to_drop, axis=1)
y = df['DIABETES']
print("Successfully loaded data and removed all Glucose/HbA1c-based features (Leakage Fixed).")
print(f"Final Features (X) Count: {X.shape[1]}")

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 3. Scale Features (Crucial for Logistic Regression, SVM, and KNN) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================================
# MODEL 1: LOGISTIC REGRESSION
# =========================================================================
print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

log_reg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg_model.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(f"Overall Accuracy: {accuracy_log_reg*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg))
# --- Confusion Matrix ---
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print("Confusion Matrix:")
print(cm_log_reg)


# =========================================================================
# MODEL 2: RANDOM FOREST CLASSIFIER (No Scaling Needed)
# =========================================================================
print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

dump(rf_model, "diabetes.joblib")

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Overall Accuracy: {accuracy_rf*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
# --- Confusion Matrix ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(cm_rf)


# =========================================================================
# MODEL 3: SUPPORT VECTOR CLASSIFIER (SVC)
# =========================================================================
print("\n" + "="*50)
print("MODEL 3: SUPPORT VECTOR CLASSIFIER (SVC)")
print("="*50)

# SVC is powerful but can be slow. Setting a linear kernel and balanced class weight.
# The 'break_ties=True' is used to handle multi-class predictions, though here we only have binary classes (0 or 1).
svc_model = SVC(kernel='linear', random_state=42, class_weight='balanced')
svc_model.fit(X_train_scaled, y_train)

y_pred_svc = svc_model.predict(X_test_scaled)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

print(f"Overall Accuracy: {accuracy_svc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc))
# --- Confusion Matrix ---
cm_svc = confusion_matrix(y_test, y_pred_svc)
print("Confusion Matrix:")
print(cm_svc)


# =========================================================================
# MODEL 4: K-NEAREST NEIGHBORS (KNN)
# =========================================================================
print("\n" + "="*50)
print("MODEL 4: K-NEAREST NEIGHBORS (KNN)")
print("="*50)

# Starting with K=5, a common default. This is sensitive to scaling!
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Overall Accuracy: {accuracy_knn*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
# --- Confusion Matrix ---
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:")
print(cm_knn)