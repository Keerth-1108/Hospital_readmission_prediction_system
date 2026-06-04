# Hospital Readmission Prediction System

Predicts whether a patient will be readmitted to hospital
within 30 days of discharge using Machine Learning.

## Tech Stack
Python | SQLite | Pandas | Scikit-learn | SMOTE | Seaborn

## Project Steps
1. Create SQLite database with 1,200 patient records
2. Extract insights using SQL queries
3. Exploratory Data Analysis — 3 visualizations
4. Engineer 5 clinical features including risk score
5. Handle class imbalance using SMOTE
6. Train Logistic Regression and Random Forest
7. Evaluate using Confusion Matrix and ROC Curve
8. Export reusable prediction pipeline

## Results
| Model               | ROC-AUC | Readmit Recall |
|---------------------|---------|----------------|
| Logistic Regression | 0.46    | 0.13           |
| Random Forest       | 0.49    | 0.24           |

Random Forest selected as final model —
achieved 2x better recall on high-risk patients.

## How to Run
pip install -r requirements.txt
python hospital_readmission_prediction_system.py

## Key Learnings
- Recall matters more than Accuracy in healthcare ML
- SMOTE effectively handles class imbalance
- Feature engineering improves minority class detection
- Synthetic data limits model performance —
  real EHR data would significantly improve results

