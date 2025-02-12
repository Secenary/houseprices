# Experiment Report: House Price Prediction Based on XGBoost

---

## Experiment Background

The goal of this experiment is to predict house prices (`SalePrice`) using machine learning algorithms based on provided housing data. The dataset consists of three files:  

- **`train.csv`**: Training data, containing house features and the target variable (`SalePrice`).  
- **`test.csv`**: Test data, containing features of houses for which predictions are required.  
- **`sample_submission.csv`**: Submission template, containing house IDs for the test set.  

Through feature engineering and hyperparameter tuning, we aim to build an efficient model for house price prediction and generate submission-compliant results.

---

## Experiment Steps

### 1. Data Loading

The `load_data` function loads the following data:  

- **Training Set**: Used for model training.  
- **Test Set**: Used to generate predictions.  
- **Submission Template**: Ensures the prediction format aligns with submission requirements.  

---

### 2. Data Preprocessing

In the `preprocess_data` function, basic preprocessing is applied to both training and test data:  

- Separate features (X) and the target variable (y).  
- Create a copy of the test data to preserve its original format.  

---

### 3. Feature Engineering

The `select_and_engineer_features` function includes:  

- **Numerical Feature Handling**:  
  - Fill missing values with the median.  
- **Categorical Feature Handling**:  
  - Fill missing values with the string `'Missing'` to represent missing categories.  

Selected features are:  

- Numerical: `['LotArea', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']`  
- Categorical: `['Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual']`  

These features were chosen based on empirical judgment of factors influencing house prices, such as area, year built, and neighborhood.  

---

### 4. Model Pipeline Construction

The `build_pipeline` function constructs a full machine learning pipeline, including:  

1. **Data Preprocessing**:  
   - Numerical features: Standardized using `StandardScaler`.  
   - Categorical features: One-Hot encoded using `OneHotEncoder`.  

2. **Model**:  
   - XGBoost Regressor (`XGBRegressor`), a powerful gradient-boosting-based regression algorithm.  

---

### 5. Model Training and Hyperparameter Tuning

The `train_and_tune_model` function uses `GridSearchCV` for hyperparameter tuning. The parameter grid includes:  

```python
param_grid = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 9],
    'model__min_child_weight': [1, 3, 5],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.2],
    'model__lambda': [0, 1, 2],
}
```

### 6. Experiment Results

#### Hyperparameter Tuning Results

The best parameters identified by `GridSearchCV`:

```python
Best Parameters: {'model__colsample_bytree': 0.8, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_child_weight': 1, 'model__n_estimators': 500, 'model__subsample': 0.6, 'model__gamma': 0, 'model__lambda': 0}
```

#### Test Set Predictions

The generated prediction file `submission.csv` meets submission requirements (see attachment).

#### Kaggle Evaluation Results

![](E:\test result.png)

### Experiment Analysis

#### Strengths

1. **Completeness**: The workflow—from data loading to submission generation—is comprehensive and clear.
2. **Modular Design**: Each step is independent, facilitating debugging and extension.
3. **Hyperparameter Tuning**: Improved model performance via `GridSearchCV`.

------

#### Limitations and Future Improvements

1. **Limited Feature Selection**: Only a small set of features was used. Future work could incorporate feature importance analysis.
2. **Narrow Parameter Grid**: Expanding the parameter grid (e.g., including more values for `subsample` and `colsample_bytree`) could yield better results.
3. **Simplistic Preprocessing**: Advanced feature engineering steps (e.g., discretization, interaction features) could enhance model performance.