import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# 1. 数据加载
def load_data(train_path, test_path, submission_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    submission_format = pd.read_csv(submission_path)
    return train_data, test_data, submission_format

# 2. 数据预处理
def preprocess_data(train_data, test_data):
    X = train_data.drop(columns=['SalePrice'])
    y = train_data['SalePrice']
    X_test = test_data.copy()
    
    # 对目标变量进行对数变换
    y = np.log1p(y)
    
    return X, X_test, y

# 3. 特征选择与工程
def select_and_engineer_features(X, X_test):
    numeric_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']
    categorical_features = ['Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual']
    
    for col in numeric_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
            X_test[col] = X_test[col].fillna(X_test[col].median())
    
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna('Missing')
            X_test[col] = X_test[col].fillna('Missing')

    # Label Encoding for categorical features
    le = LabelEncoder()
    for col in categorical_features:
        if col in X.columns:
            X[col] = le.fit_transform(X[col].astype(str))  # 转换为字符串进行编码
            X_test[col] = le.transform(X_test[col].astype(str))  # 对测试集进行相同的编码处理
    
    return numeric_features, categorical_features

# 4. 构建管道
def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features),
                      ('cat', categorical_transformer, categorical_features)]
    )
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=1)  # 设置 verbose = 1 来显示进度
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    return pipeline

# 5. 模型训练与调参
def train_and_tune_model(pipeline, X_train, y_train):
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
    
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)  # 设置 verbose = 2 输出详细信息
    search.fit(X_train, y_train)
    
    print(f"最佳参数: {search.best_params_}")
    return search.best_estimator_

# 6. 生成预测提交文件
def generate_submission(pipeline, X_test, submission_format, output_path):
    predictions = pipeline.predict(X_test)
    predictions = np.expm1(predictions)  # 对数反变换
    submission_format['SalePrice'] = predictions
    submission_format = submission_format.sort_values(by='Id')
    submission_format.to_csv(output_path, index=False)

# 主程序
def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    submission_path = 'sample_submission.csv'
    
    train_data, test_data, submission_format = load_data(train_path, test_path, submission_path)
    X, X_test, y = preprocess_data(train_data, test_data)
    numeric_features, categorical_features = select_and_engineer_features(X, X_test)
    
    pipeline = build_pipeline(numeric_features, categorical_features)
    
    pipeline = train_and_tune_model(pipeline, X, y)
    
    generate_submission(pipeline, X_test, submission_format, './submission.csv')

if __name__ == "__main__":
    main()
