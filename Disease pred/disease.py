import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
df = pd.read_csv(r"C:\Users\manas\OneDrive\Documents\MERN\ML projects\Disease pred\Training.csv", index_col=0) 
df_test = pd.read_csv(r"C:\Users\manas\OneDrive\Documents\MERN\ML projects\Disease pred\Testing.csv", index_col=0)
# Initial Data Exploration
print(df_test.head())
print(df.head())
print(df.info())
print(df.shape)
print(df.isna().sum())
print(df.nunique())
print("Data Preprocessing and Visualization")
disease_counts = df['prognosis'].value_counts()
plt.figure(figsize=(12, 8))
bars = plt.bar(disease_counts.index, disease_counts.values, color='skyblue')
plt.xticks(rotation=90)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=8)
plt.title("Frequency of Each Disease")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Drop the unnamed last column (if exists)
df = df.drop(df.columns[-1], axis=1)
# Encode labels
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])
df_test['prognosis'] = le.transform(df_test['prognosis'])
# Split features and labels
X_train = df.drop('prognosis', axis=1)
y_train = df['prognosis']
X_test = df_test.drop('prognosis', axis=1)
y_test = df_test['prognosis']
print("Random forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf_train = rf.predict(X_train)  
y_pred_rf_test = rf.predict(X_test)    
# Training Metrics
print("\n Training Metrics:")
print("Training Accuracy:", accuracy_score(y_train, y_pred_rf_train))
print("Training Precision:", precision_score(y_train, y_pred_rf_train, average='weighted'))
print("Training Recall:", recall_score(y_train, y_pred_rf_train, average='weighted'))
print("Training F1 Score:", f1_score(y_train, y_pred_rf_train, average='weighted'))
# Testing Metrics
print("\n Testing Metrics:")
print("Testing Accuracy:", accuracy_score(y_test, y_pred_rf_test))
print("Testing Precision:", precision_score(y_test, y_pred_rf_test, average='weighted'))
print("Testing Recall:", recall_score(y_test, y_pred_rf_test, average='weighted'))
print("Testing F1 Score:", f1_score(y_test, y_pred_rf_test, average='weighted'))
# Classification Reports
print("\nClassification Report for Training Data:")
print(classification_report(y_train, y_pred_rf_train, target_names=le.classes_))

print("\nClassification Report for Testing Data:")
print(classification_report(y_test, y_pred_rf_test, target_names=le.classes_))
# Confusion Matrix for Test Data
cm = confusion_matrix(y_test, y_pred_rf_test)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
