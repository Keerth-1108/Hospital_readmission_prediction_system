# Install & Import Required Libraries
!pip install imbalanced-learn      #Fixes unequal class distribution

import sqlite3                     # Connect to the database
import pandas as pd                # Work with tables
import numpy as np                 # Generate random data / math
import matplotlib.pyplot as plt    # Plot graphs
import seaborn as sns              # Better-looking plots
import joblib                      # Save/load models

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling  import SMOTE # handle class imbalance

#Connect to the Database
conn   = sqlite3.connect('hospital.db')
cursor = conn.cursor()

#Create the patients table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        age                   INTEGER,
        gender                TEXT,
        diagnosis_code        TEXT,
        num_medications       INTEGER,
        num_lab_procedures    INTEGER,
        num_procedures        INTEGER,
        length_of_stay        INTEGER,
        num_prior_admissions  INTEGER,
        discharge_disposition TEXT,
        readmitted            INTEGER
    )
''')
conn.commit()

#GeneratE 1,200 Synthetic Patient Records
np.random.seed(42)
N = 1200
diagnoses    = ['N18','J18','K57','E11','I10','Z79','J44','I50']
dispositions = ['Home','SNF','AMA','Rehab','HospiceHome']

#N18 - Chronic Kidney Disease (CKD)
#J18 - Pneumonia
#K57 - Diverticular Disease (intestine)
#E11 - Type 2 Diabetes
#I10 - Hypertension (High Blood Pressure)
#Z79 - Long-term medication use
#J44 - COPD (Chronic lung disease)
#I50 - Heart Failure


records = pd.DataFrame({
    'age'                   : np.random.randint(18, 92, N),
    'gender'                : np.random.choice(['Male','Female'], N),
    'diagnosis_code'        : np.random.choice(diagnoses, N),
    'num_medications'       : np.random.randint(1, 30, N),
    'num_lab_procedures'    : np.random.randint(0, 65, N),
    'num_procedures'        : np.random.randint(0, 8, N),
    'length_of_stay'        : np.random.randint(1, 18, N),
    'num_prior_admissions'  : np.random.randint(0, 6, N),
    'discharge_disposition' : np.random.choice(dispositions, N),
    'readmitted'            : np.random.choice([0,1], N, p=[0.68, 0.32]),
})

#Insert Data into the SQL Table
records.to_sql('patients', conn, if_exists='append', index=False)

#Load all records
df = pd.read_sql_query('SELECT * FROM patients', conn)
print(df.shape)
print(df.head())

#Query 1 — Find Readmission Rate by Discharge Type
q1 = pd.read_sql_query('''
    SELECT discharge_disposition,
    COUNT(*) AS total_patients,
    SUM(readmitted) AS readmitted_count,
    ROUND(100.0 * SUM(readmitted)/COUNT(*), 2) AS readmit_rate_pct
    FROM   patients GROUP  BY discharge_disposition ORDER  BY readmit_rate_pct DESC
''', conn)
print(q1)
print()

#Query 2 — Identify High-risk seniors with multiple prior admissions
q2 = pd.read_sql_query('''
    SELECT patient_id, age, num_prior_admissions, length_of_stay, readmitted
    FROM   patients WHERE  age > 65 AND num_prior_admissions >= 3 ORDER  BY num_prior_admissions DESC LIMIT  20
''', conn)
print(q2)
print()
#Query 3 — Calculate Avg Stay &  Avg Medications Per Diagnosis
q3 = pd.read_sql_query('''
    SELECT diagnosis_code,
    ROUND(AVG(length_of_stay), 2)  AS avg_los,
    ROUND(AVG(num_medications), 2) AS avg_meds,
    COUNT(*) AS patient_count
    FROM   patients GROUP  BY diagnosis_code ORDER  BY avg_los DESC
''', conn)
print(q3)

#Data quality check
print(df.info())
print()
print(df.isnull().sum())
print()
print(df.describe())
print()
print(df['readmitted'].value_counts())

#Age Distribution by Readmission Status (Histogram)
plt.figure(figsize=(9,4))
sns.histplot(data=df, x='age', hue='readmitted',kde=True, bins=30, palette='Set1',multiple='dodge')
plt.title('Age Distribution by Readmission Status')
plt.show()
print()

#Box Plot: Length of Stay vs Readmission (Box Plot)
plt.figure(figsize=(7,4))
sns.boxplot(data=df, hue='readmitted', y='length_of_stay', palette='Blues')
plt.title('Length of Stay vs Readmission')
plt.show()
print()

#Correlation heatmap
num_cols = df.select_dtypes(include='number').drop(columns=['patient_id','readmitted'])
plt.figure(figsize=(7,4))
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f',cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

#Create Clinically Meaningful Features
df['risk_score'] = (df['num_prior_admissions'] * 3.0 + df['length_of_stay'] * 0.6 + df['num_medications'] * 0.4)
df['is_senior'] = (df['age'] >= 65).astype(int)
df['high_med_burden'] = (df['num_medications'] > 15).astype(int)

chronic_codes = ['E11','I10','N18','I50','J44']

df['is_chronic'] = df['diagnosis_code'].isin(chronic_codes).astype(int)
df['high_lab_load'] = (df['num_lab_procedures'] > 40).astype(int)
print(df[['risk_score','is_senior','high_med_burden','is_chronic']].head())

# Encode Categorical Variables
le_gender   = LabelEncoder()
le_discharge= LabelEncoder()
le_diag     = LabelEncoder()
df['gender_enc']    = le_gender.fit_transform(df['gender'])
df['discharge_enc'] = le_discharge.fit_transform(df['discharge_disposition'])
df['diagnosis_enc'] = le_diag.fit_transform(df['diagnosis_code'])

#Verifying the Mapping
print('Gender classes:    ', le_gender.classes_)
print('Discharge classes: ', le_discharge.classes_)
print('Diagnosis classes: ', le_diag.classes_)

#Define feature columns
FEATURES = [
    'age', 'gender_enc', 'diagnosis_enc',
    'num_medications', 'num_lab_procedures', 'num_procedures',
    'length_of_stay', 'num_prior_admissions', 'discharge_enc',
    'risk_score', 'is_senior', 'high_med_burden',
    'is_chronic', 'high_lab_load'
]
X = df[FEATURES]
y = df['readmitted']

print('Class distribution BEFORE balancing:')
print(y.value_counts(normalize=True).round(3))

#Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#SMOTE: Synthetic Minority Oversampling (only on TRAINING data)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print('\nClass distribution AFTER SMOTE:')
print(pd.Series(y_train_bal).value_counts())

#Scale Features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

#Train Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
lr.fit(X_train_sc, y_train_bal)

y_pred_lr   = lr.predict(X_test_sc)
y_proba_lr  = lr.predict_proba(X_test_sc)[:, 1]

print('Logistic Regression')
print(classification_report(y_test, y_pred_lr, target_names=['No Readmit','Readmitted']))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba_lr):.4f}')

#Train Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators  = 200,
    max_depth     = 10,
    min_samples_split = 4,
    class_weight  = 'balanced',
    random_state  = 42,
    n_jobs        = -1
)
rf.fit(X_train_bal, y_train_bal)

y_pred_rf   = rf.predict(X_test)
y_proba_rf  = rf.predict_proba(X_test)[:, 1]

print('Random Fores')
print(classification_report(y_test, y_pred_rf, target_names=['No Readmit','Readmitted']))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba_rf):.4f}')

#Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(7, 4))
for ax, preds, name in zip(axes, [y_pred_lr, y_pred_rf], ['Logistic Regression','Random Forest']):
  cm = confusion_matrix(y_test, preds)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,xticklabels=['No Readmit','Readmitted'],yticklabels=['No Readmit','Readmitted'])
  ax.set_title(f'{name}\nConfusion Matrix', fontsize=12)
  ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.show()

#ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_rf = roc_auc_score(y_test, y_proba_rf)

plt.figure(figsize=(7, 4))
plt.plot(fpr_lr, tpr_lr, 'b-',  label=f'Logistic Regression (AUC={auc_lr:.3f})')
print()
plt.plot(fpr_rf, tpr_rf, 'g--', label=f'Random Forest(AUC={auc_rf:.3f})')
plt.plot([0,1], [0,1], 'k:', linewidth=1)
print()
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Model Comparison')
plt.legend(loc='lower right')
plt.show()

#Feature Importance & Model Interpretation
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(9, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score (Mean Decrease in Impurity)')
plt.tight_layout()
plt.show()

print(importances.sort_values(ascending=False).head(5))
print()
coeff_df = pd.DataFrame({
    'Feature'    : FEATURES,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(coeff_df)

#Save model artifacts
joblib.dump(rf,'readmission_rf_model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(le_discharge,'le_discharge.pkl')
joblib.dump(le_diag,'le_diag.pkl')
print('Model artifacts saved successfully.')

#Prediction function for new patients
def predict_readmission(patient_dict):
    """
    patient_dict: dict with keys matching FEATURES
    Returns: probability of 30-day readmission
    """
    model  = joblib.load('readmission_rf_model.pkl')
    new_df = pd.DataFrame([patient_dict])[FEATURES]
    proba  = model.predict_proba(new_df)[0][1]
    risk   = 'HIGH RISK' if proba > 0.5 else 'LOW RISK'
    return proba, risk
#Example new patient
patient = {
    'age':72, 'gender_enc':1, 'diagnosis_enc':0,
    'num_medications':22, 'num_lab_procedures':48,
    'num_procedures':3, 'length_of_stay':11,
    'num_prior_admissions':4, 'discharge_enc':2,
    'risk_score': 4*3 + 11*0.6 + 22*0.4,
    'is_senior':1, 'high_med_burden':1, 'is_chronic':1, 'high_lab_load':1
}

prob, category = predict_readmission(patient)
print(f'Readmission Probability: {prob:.1%}  |  Status: {category}')
