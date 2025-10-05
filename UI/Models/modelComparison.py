# Importing the required libraries 
import time 
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# Function to measure execution time
def measure_time(model, X_train, y_train, X_test, y_test):
    
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    fit_time = time.time() - start_time
    
    start_time = time.time()
    
    predictions = model.predict(X_test)
    
    pred_time = time.time() - start_time
    
    return fit_time, pred_time, predictions


# Loading and cleaning dataset 
dataset = pd.read_csv('sensorData.csv')

# Removing irrelevant columns
dataset.columns = dataset.columns.str.strip()
dataset.drop(columns=['Timestamp'], inplace=True, errors='ignore')

# Handling missing values
dataset.fillna(dataset.mean(), inplace=True)

# Selecting relevant features
X = dataset[['Temperature', 'Humidity', 'Gas']]

# Target variable (Anomaly column)
y = dataset['Anomaly']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
   
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest 
rf_model = RandomForestClassifier(random_state=42)
rf_fit_time, rf_pred_time, rf_pred = measure_time(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Model 2: XGBoost 
xgb_model = XGBClassifier(random_state=42)
xgb_fit_time, xgb_pred_time, xgb_pred = measure_time(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Model 3: Isolation Forest 
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train_scaled)  # Fit only on normal data
iso_pred = iso_forest.predict(X_test_scaled)

# Converting Isolation Forest predictions: 1 -> 0 (normal), -1 -> 1 (anomaly)
iso_pred = (iso_pred == -1).astype(int)  # Now, 1 = anomaly, 0 = normal


# Evaluation: Calculating performance metrics 
def evaluate_performance(y_true, y_pred, model_name):
   
    print(f"--- {model_name} ---")
   
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], labels=[0, 1]))
   
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    
    # ROC AUC
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC AUC: {auc:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Evaluating each model
evaluate_performance(y_test, rf_pred, "Random Forest")
evaluate_performance(y_test, xgb_pred, "XGBoost")
evaluate_performance(y_test, iso_pred, "Isolation Forest")