
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from imblearn.over_sampling import SMOTE  # For oversampling
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('/Users/pavithrasenthilkumar/Documents/School/Enterprise Business Analytics/GCP Pipeline Project/readmission_prediction/Data/healthcare_readmissions_dataset_train.csv')


data=data.drop(['PatientID'], axis=1)

### Imputing the number of visits and medications both with Median

median_visits = data['Number of Prior Visits'].median()
data['Number of Prior Visits'].fillna(median_visits, inplace=True)

# Impute 'medication_type' with median
median_medication = data['Medications Prescribed'].median()
data['Medications Prescribed'].fillna(median_medication, inplace=True)


data['Exercise Frequency'].fillna('No exercise', inplace=True)

data['Type of Treatment'].fillna('No Treatment', inplace=True)

## Dropping these from the data since they hare related to BMI
data = data.drop(['Weight (kg)', 'Adjusted Weight (kg)'], axis=1)


data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Convert 'smoker' to binary: 'Yes' -> 1, 'No' -> 0
data['Smoker'] = data['Smoker'].map({True: 1, False: 0})


data=data.drop(['Hospital ID'],axis=1)

data_encoded = pd.get_dummies(data, columns=['Diet Type', 'Type of Treatment', 'Ethnicity', 'Exercise Frequency'])

# Convert True/False to 1/0
data_encoded = data_encoded.astype(int)

data_encoded= data_encoded.drop(['Height (m)'], axis=1)


X = data_encoded.drop('Readmission within 30 Days', axis=1)  # Features
y = data_encoded['Readmission within 30 Days']  # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")

model = LogisticRegression()
model.fit(X_train_scaled, y_train_resampled)
print("heyyyy")

joblib.dump(model, 'model.pkl')
print("Model saved successfully.")

print("Orinigal data before predict ",data.head(5))

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average='binary')  # You can use 'binary' for binary classification or 'macro', 'micro', 'weighted' for multi-class
print(f'F1 Score: {f1:.4f}')

#### Testing Data

train_data = pd.read_csv('/Users/pavithrasenthilkumar/Documents/School/Enterprise Business Analytics/GCP Pipeline Project/readmission_prediction/Data/healthcare_readmissions_dataset_test.csv')
patient_ids = train_data['PatientID']

median_visits = train_data['Number of Prior Visits'].median()
train_data['Number of Prior Visits'].fillna(median_visits, inplace=True)

# Impute 'medication_type' with median
median_medication = train_data['Medications Prescribed'].median()
train_data['Medications Prescribed'].fillna(median_medication, inplace=True)

train_data['Exercise Frequency'].fillna('No exercise', inplace=True)

train_data['Type of Treatment'].fillna('No Treatment', inplace=True)

## Dropping these from the data since they hare related to BMI
train_data = train_data.drop(['Weight (kg)', 'Adjusted Weight (kg)'], axis=1)

train_data['Gender'] = train_data['Gender'].map({'Male': 1, 'Female': 0})

# Convert 'smoker' to binary: 'Yes' -> 1, 'No' -> 0
train_data['Smoker'] = train_data['Smoker'].map({True: 1, False: 0})

train_data=train_data.drop(['Hospital ID','PatientID'],axis=1)

train_data_encoded = pd.get_dummies(train_data, columns=['Diet Type', 'Type of Treatment', 'Ethnicity', 'Exercise Frequency'])

# Convert True/False to 1/0
train_data_encoded = train_data_encoded.astype(int)

train_data_encoded= train_data_encoded.drop(['Height (m)'], axis=1)

train_data_scaled = scaler.transform(train_data_encoded)
print("tarin data ", train_data.head(5))


predictions = model.predict(train_data_scaled)

# Combine predictions with Patient IDs
results = pd.DataFrame({
    'PatientID': patient_ids,
    'Readmission Prediction': predictions
})

# Save predictions to a CSV file
results.to_csv('predictions.csv', index=False)

print('Predictions saved to /path_to_save_predictions.csv')





