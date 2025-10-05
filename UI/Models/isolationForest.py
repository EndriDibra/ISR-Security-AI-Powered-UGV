# Importing the required libraries 
import warnings
import pandas as pd
import seaborn as sb
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


# Ignoring warnings
warnings.filterwarnings('ignore')

# Loading the sensor dataset
dataset = pd.read_csv('sensorData.csv')

# Removing spaces in column names
dataset.columns = dataset.columns.str.strip()

# Dropping irrelevant columns
if 'Timestamp' in dataset.columns:
    
    dataset.drop(columns=['Timestamp'], inplace=True)

# Removing duplicate rows, if any
dataset.drop_duplicates(inplace=True)

# Displaying dataset sample and info
print("Dataset Sample:\n", dataset.head())
print("\nDataset Shape:", dataset.shape)
print("\nMissing values:\n", dataset.isnull().sum())

# Filling missing values only in numeric columns
dataset.fillna(dataset.select_dtypes(include=['number']).mean(), inplace=True)

# Displaying cleaned dataset
print("\nDataset after cleaning process:\n", dataset.head())

# Selecting relevant features
X = dataset[['Temperature', 'Humidity', 'Gas']]

# Target variable (Anomaly column)
y = dataset['Anomaly']

# Normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Defining the model for IsolationForest
model = IsolationForest(random_state=42)

# Creating a parameter grid for tuning contamination
param_grid = {
    
    'contamination': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
}


# Custom scoring function that prioritizes recall for anomalies
def custom_scorer(estimator, X, y):

    y_pred = estimator.predict(X)

    # Calculate recall for anomalies (class == -1)
    recall_anomaly = recall_score(y, y_pred, pos_label=-1)

    return recall_anomaly


# Defining the stratified cross-validation strategy
stratified_kfold = StratifiedKFold(n_splits=5)

# Using GridSearchCV with custom scoring and StratifiedKFold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring=make_scorer(custom_scorer))
grid_search.fit(X_scaled, y) 

# Best contamination found
best_contamination = grid_search.best_params_['contamination']
print(f"\nBest Contamination Value (based on recall for anomalies): {best_contamination}")

# Fitting the model with the best contamination value
model = IsolationForest(contamination=best_contamination, random_state=42)
model.fit(X_scaled)

# Saving the trained model and scaler for use in the server
dump(model, 'iso_Model.joblib')
dump(scaler, 'iso_Scaler.joblib')

# Predicting anomalies: -1 = anomaly, 1 = normal
dataset['Predicted'] = model.predict(X_scaled)
dataset['Predicted_Label'] = (dataset['Predicted'] == -1).astype(int)  # 1 = anomaly

# Anomaly score (lower = more anomalous)
dataset['Anomaly_Score'] = model.decision_function(X_scaled)

# Plot anomaly score distribution
plt.figure(figsize=(8, 4))
sb.histplot(dataset['Anomaly_Score'], bins=50, kde=True)
plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("Anomaly Score")
plt.tight_layout()
plt.show()

# Evaluating model against actual 'Anomaly' column
y_true = dataset['Anomaly']
y_pred = dataset['Predicted_Label']

print("\n--- Evaluation Report ---")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Visualizing true vs predicted anomalies with breakdown
plt.figure(figsize=(10, 6))

tp = dataset[(y_pred == 1) & (y_true == 1)]
fp = dataset[(y_pred == 1) & (y_true == 0)]
fn = dataset[(y_pred == 0) & (y_true == 1)]

plt.scatter(tp['Temperature'], tp['Gas'], c='green', label='True Positive', marker='o', s=60)
plt.scatter(fp['Temperature'], fp['Gas'], c='orange', label='False Positive', marker='x', s=60)
plt.scatter(fn['Temperature'], fn['Gas'], c='red', label='False Negative', marker='^', s=60)

plt.xlabel("Temperature")
plt.ylabel("Gas")
plt.title("Anomaly Detection: TP vs FP vs FN")
plt.legend()
plt.tight_layout()
plt.show()

# Visualizing anomalies with Temperature, Humidity, and Gas

# Plot: Temperature vs Gas (with anomalies marked)
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Temperature'], dataset['Gas'], c=dataset['Predicted_Label'], cmap='coolwarm', s=40)
plt.xlabel('Temperature')
plt.ylabel('Gas')
plt.title('Temperature vs Gas (with Anomalies)')
plt.colorbar(label='Anomaly (0 = Normal, 1 = Anomaly)')
plt.tight_layout()
plt.show()

# Plot: Temperature vs Humidity (with anomalies marked)
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Temperature'], dataset['Humidity'], c=dataset['Predicted_Label'], cmap='coolwarm', s=40)
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Temperature vs Humidity (with Anomalies)')
plt.colorbar(label='Anomaly (0 = Normal, 1 = Anomaly)')
plt.tight_layout()
plt.show()

# Plot: Humidity vs Gas (with anomalies marked)
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Humidity'], dataset['Gas'], c=dataset['Predicted_Label'], cmap='coolwarm', s=40)
plt.xlabel('Humidity')
plt.ylabel('Gas')
plt.title('Humidity vs Gas (with Anomalies)')
plt.colorbar(label='Anomaly (0 = Normal, 1 = Anomaly)')
plt.tight_layout()
plt.show()


# Function to detect anomaly for new input
def detect_anomaly(temperature, humidity, gas):

    input_scaled = scaler.transform([[temperature, humidity, gas]])

    prediction = model.predict(input_scaled)[0]

    score = model.decision_function(input_scaled)[0]

    label = "Anomaly Detected (Smoke/Fire Possible)" if prediction == -1 else "Normal"

    return f"{label} | Anomaly Score: {score:.4f}"


# Example predictions
print("\nExample Prediction 1:")
print(detect_anomaly(25.0, 60.0, 100))  # likely normal

print("\nExample Prediction 2:")
print(detect_anomaly(40.0, 40.0, 350))  # possibly fire/smoke