# Mental Health Risk Prediction - Machine Learning Project

## 📋 Project Overview

This project develops a comprehensive machine learning model to predict mental health risk levels based on demographic, lifestyle, and medical history factors. The model achieves **100% accuracy** using optimized feature selection and hyperparameter tuning, with Logistic Regression emerging as the best performer.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Logistic Regression |
| **Test Accuracy** | 100.00% |
| **F1-Score** | 1.0000 |
| **AUC Score** | 1.0000 |
| **Balanced Accuracy** | 100.00% |

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Total Records | 45,858 |
| Original Features | 16 |
| Features After Reduction | 15 |
| Target Classes | Low Risk (0) / High Risk (1) |
| Class Distribution (Low Risk) | 57.61% |
| Class Distribution (High Risk) | 42.39% |

---

## 🔬 Feature Engineering

### Risk Score Calculation

The target variable `Mental_Health_Risk` was created using a weighted risk score:

| Risk Factor | Weight |
|-------------|--------|
| History of Mental Illness | 3 |
| Family History of Depression | 2 |
| History of Substance Abuse | 2 |
| High Alcohol Consumption | 2 |
| Poor Sleep Patterns | 2 |
| Unhealthy Dietary Habits | 1 |
| Sedentary Lifestyle | 1 |
| Current Smoking | 1 |
| Chronic Medical Conditions | 1 |
| Former Smoking | 0.5 |
| Moderate Alcohol Consumption | 1 |
| Fair Sleep Patterns | 1 |

### Feature Reduction (Based on Correlation Matrix)

Highly correlated features were removed to avoid multicollinearity:

| Removed Feature | Correlation | Reason |
|----------------|-------------|--------|
| Marital Status_Single | -0.60 (with Age) | High negative correlation |
| Marital Status_Widowed | -0.48 (with Age) | High negative correlation |

### Final Feature Set (15 Features)

| Feature Type | Features |
|--------------|----------|
| **Demographic** | Age, Education Level, Number of Children |
| **Lifestyle** | Smoking Status, Physical Activity Level, Alcohol Consumption, Dietary Habits, Sleep Patterns |
| **Socioeconomic** | Income, Marital Status_Married, Employment Status_Unemployed |
| **Medical History** | History of Mental Illness_Yes, History of Substance Abuse_Yes, Family History of Depression_Yes, Chronic Medical Conditions_Yes |

---

## 🤖 Models Evaluated

| Model | Test Accuracy | Balanced Accuracy | F1-Score | AUC | MCC |
|-------|--------------|-------------------|----------|-----|-----|
| **Logistic Regression** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| MLP Classifier | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 99.97% | 99.97% | 0.9997 | 1.0000 | 0.9994 |
| Gradient Boost | 99.86% | 99.86% | 0.9986 | 0.9999 | 0.9972 |
| Random Forest | 99.24% | 99.24% | 0.9924 | 0.9997 | 0.9849 |
| Decision Tree | 97.85% | 97.85% | 0.9787 | 0.9965 | 0.9571 |

---

## 🏆 Top Risk Factors (Logistic Regression Coefficients)

| Rank | Feature | Coefficient Magnitude |
|------|---------|----------------------|
| 1 | History of Mental Illness_Yes | 8.3996 |
| 2 | History of Substance Abuse_Yes | 5.5535 |
| 3 | Family History of Depression_Yes | 5.3458 |
| 4 | Alcohol Consumption | 4.2965 |
| 5 | Sleep Patterns | 4.0946 |
| 6 | Chronic Medical Conditions_Yes | 2.7741 |
| 7 | Physical Activity Level | 2.1513 |
| 8 | Smoking Status | 2.0318 |
| 9 | Dietary Habits | 1.9117 |
| 10 | Education Level | 0.0464 |

---

## 🔧 Hyperparameter Tuning Results

### Logistic Regression (Best Model)
```python
{
    'C': 0.1,
    'penalty': 'l2',
    'solver': 'lbfgs'
}
