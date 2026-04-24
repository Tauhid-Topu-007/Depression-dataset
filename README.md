# Mental Health Risk Prediction Model

## Project Overview

This project develops a comprehensive machine learning model to predict mental health risk levels based on demographic, lifestyle, and medical history factors. The model achieves **99.99% accuracy** using various ML algorithms, with XGBoost emerging as the best performer.

## Key Features

- **Dataset Size**: 413,768 records
- **Input Features**: 17 risk factors including:
  - Demographics (Age, Marital Status, Education Level)
  - Lifestyle factors (Smoking, Physical Activity, Alcohol Consumption)
  - Medical history (Mental Illness, Substance Abuse, Family History)
  - Socioeconomic factors (Employment Status, Income)

- **Target Variable**: Binary classification (High Risk / Low Risk)

## Model Performance

| Model | Test Accuracy | F1-Score | AUC | MCC |
|-------|--------------|----------|-----|-----|
| **XGBoost** | 99.99% | 99.99% | 1.0000 | 0.9997 |
| Logistic Regression | 99.98% | 99.98% | 0.9999 | 0.9997 |
| Random Forest | 99.94% | 99.94% | 0.9999 | 0.9987 |
| MLP Classifier | 99.98% | 99.98% | 0.9999 | 0.9997 |

## Top Risk Factors

1. **History of Mental Illness** (34.5% importance)
2. **History of Substance Abuse** (12.8% importance)
3. **Family History of Depression** (11.7% importance)
4. **Alcohol Consumption** (8.7% importance)
5. **Sleep Patterns** (6.6% importance)

## Technical Implementation

### Data Preprocessing
- Missing value handling (mode for categorical, median for numerical)
- Outlier detection and capping using IQR method
- Feature engineering (Risk Score calculation)
- Ordinal encoding for ordered categories
- One-hot encoding for nominal categories

### Handling Class Imbalance
- Initial distribution: 57.7% Low Risk / 42.3% High Risk
- SMOTE oversampling to balance classes
- Class weight adjustment for imbalanced handling

### Feature Selection
- Statistical significance testing (p-value < 0.05)
- Point-biserial correlation for continuous features
- Chi-square tests for categorical features
- All 17 features found statistically significant

### Models Evaluated
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- MLP Neural Network

### Hyperparameter Tuning
- RandomizedSearchCV with 5-fold cross-validation
- Balanced accuracy as scoring metric
- Best XGBoost parameters:
  ```python
  {
      'subsample': 0.9,
      'n_estimators': 300,
      'max_depth': 7,
      'learning_rate': 0.1,
      'gamma': 0.1,
      'colsample_bytree': 0.9
  }
