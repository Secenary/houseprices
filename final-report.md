# 实验报告：基于 XGBoost 的房价预测

---

## 实验背景

本实验的目标是基于提供的房屋数据，利用机器学习算法预测房价（`SalePrice`）。实验数据由三个文件组成：

- **`train.csv`**：训练数据，包含房屋的特征和目标变量（`SalePrice`）。  
- **`test.csv`**：测试数据，包含需要预测的房屋特征。  
- **`sample_submission.csv`**：提交文件的模板，包含测试集的房屋 ID。  

通过特征工程和超参数调优，我们希望构建一个能够高效预测房价的模型，并生成符合提交格式的预测结果。

---

## 实验步骤

### 1. 数据加载

使用 `load_data` 函数加载数据，包括：

- **训练集**：用于模型训练。
- **测试集**：用于生成预测结果。
- **提交模板**：确保预测结果格式与提交要求一致。

---

### 2. 数据预处理

在 `preprocess_data` 函数中，对训练数据和测试数据进行基本预处理：

- 分离特征（X）和目标变量（y）。
- 复制测试数据，保持原数据格式。

---

### 3. 特征工程

在 `select_and_engineer_features` 函数中完成：

- **数值特征处理**：
  - 填充缺失值，使用中位数填充。
- **类别特征处理**：
  - 填充缺失值，用字符串 `'Missing'` 表示缺失类别。

选择了以下特征：

- 数值特征：`['LotArea', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']`  
- 类别特征：`['Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual']`  

这些特征的选择基于对房价影响因素的经验判断，例如房屋面积、年份和周边社区。

---

### 4. 构建模型管道

在 `build_pipeline` 函数中，构建了一个完整的机器学习管道，包括：

1. **数据预处理**：
   - 数值特征：标准化处理（`StandardScaler`）。
   - 类别特征：One-Hot 编码（`OneHotEncoder`）。

2. **模型**：
   - 使用 XGBoost 回归器（`XGBRegressor`），一个基于梯度提升的强大回归算法。

---

### 5. 模型训练与超参数调优

在 `train_and_tune_model` 函数中，通过 `GridSearchCV` 对模型进行超参数调优。调整了更详细的参数网格，包括：

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

### 6.实验结果

#### 模型调优结果

通过 `GridSearchCV` 超参数调优，找到最佳参数组合：

```
最佳参数: {'model__colsample_bytree': 0.8, 'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__min_child_weight': 1, 'model__n_estimators': 500, 'model__subsample': 0.6, 'model__gamma' : 0, 'model__lambda' : 0}
```

#### 测试集预测结果

生成的预测文件 `submission.csv` 符合提交要求。(见附件)

#### Kaggle网站测试结果

![alt text](image.png)

### 实验分析

#### 优点

1. **完整性**：从数据加载到提交文件生成，流程完整且清晰。
2. **模块化设计**：各步骤独立，便于调试和扩展。
3. **参数调优**：通过 `GridSearchCV` 寻找最佳参数，提高了模型性能。

------

#### 不足与改进方向

1. **特征选择有限**： 当前只选择了少量特征，未来可以通过特征重要性分析增加有意义的特征。
2. **参数网格有限**： 目前的参数网格范围较窄，可以增加其他参数如 `subsample` 和 `colsample_bytree`。
3. **数据预处理简单**： 可以加入更多特征工程步骤，例如离散化、交互特征生成等。