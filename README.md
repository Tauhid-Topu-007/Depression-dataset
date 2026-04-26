# Depression Risk Prediction Model

## 📋 Project Overview

This project develops a machine learning model to predict depression risk using socio-demographic and lifestyle factors. The model achieves **80.7% balanced accuracy**, representing a **21.8% improvement** over the baseline model through advanced feature engineering, ensemble methods, and optimization techniques.

## 🎯 Key Achievements

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Balanced Accuracy | 58.9% | **80.7%** | **+21.8%** |
| AUC Score | 0.611 | **0.884** | **+0.273** |
| F1-Score | 0.491 | **0.791** | **+0.300** |
| MCC | 0.193 | **0.610** | **+0.417** |

## 🗂️ Dataset

The dataset contains **18,345** records with **16** features including:

- **Demographics**: Age, Marital Status, Education Level
- **Lifestyle**: Smoking Status, Physical Activity, Alcohol Consumption
- **Socioeconomic**: Income, Employment Status
- **Health History**: Mental Illness, Substance Abuse, Chronic Medical Conditions

**Note**: Target variable was synthetically created from clinical history features for demonstration purposes.

## 🚀 Methodology

### 1. Data Preprocessing
- Ordinal encoding for categorical variables (Education, Physical Activity, etc.)
- One-hot encoding for nominal features (Marital Status, Employment Status)
- Outlier capping using 3×IQR method
- Variance threshold filtering (0.01 threshold)

### 2. Feature Engineering
- **Polynomial features** for top predictive variables
- **Interaction terms** between key features
- **Domain-specific features**:
  - Age² and Age_Log transformations
  - Combined risk scores
  - Children grouping and indicators

### 3. Handling Class Imbalance
- Original dataset: 11.2:1 imbalance ratio
- **Borderline-SMOTE** sampling (best performer)
- SMOTE-Tomek for combined sampling

### 4. Models Tested

| Model | Balanced Accuracy | AUC |
|-------|------------------|-----|
| Logistic Regression | 58.9% | 0.611 |
| Decision Tree | 59.4% | 0.627 |
| Gradient Boost | 60.5% | 0.649 |
| XGBoost | 50.0% | 0.645 |
| MLP Classifier | 60.1% | 0.641 |
| **Random Forest (Tuned)** | **67.6%** | **0.749** |
| Voting Classifier | 78.8% | 0.874 |
| Stacking Classifier | 80.2% | 0.879 |
| **Random Forest (Advanced)** | **80.7%** | **0.884** |

### 5. Advanced Techniques
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV
- **Feature Selection**: RFECV with Random Forest
- **Ensemble Methods**:
  - Soft Voting Classifier (RF, GB, XGB)
  - Stacking Classifier with Logistic Regression meta-learner
- **Deep Learning**: TensorFlow neural network with:
  - Batch normalization
  - Dropout regularization (0.3, 0.2, 0.1)
  - Early stopping & ReduceLROnPlateau callbacks

## 📊 Best Model Details

**Random Forest Tuned** (Best performing)

### Hyperparameters:
```python
{
    'n_estimators': 300,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': None,
    'bootstrap': False,
    'class_weight': 'balanced'
}```
