import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./TrainDataset2024.csv')
data_test = pd.read_excel('./FinalTestDataset2024.xls')
def data_preprocess(data):
    
    missing_mask = data.select_dtypes(include=[float, int]) == 999
    temp_imputer = SimpleImputer(missing_values=999, strategy='mean')
    data_temp_filled = temp_imputer.fit_transform(data.select_dtypes(include=[float, int]))

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_temp_filled)

    data_for_knn = data_normalized.copy()
    data_for_knn[missing_mask] = 999
    knn_imputer = KNNImputer(missing_values=999, n_neighbors=6)
    knn_imputer.fit(data_normalized)
    data_imputed = knn_imputer.transform(data_for_knn)

    data_imputed_df = pd.DataFrame(scaler.inverse_transform(data_imputed),
                                columns=data.select_dtypes(include=[float, int]).columns)
    data_imputed_df


    categorical_cols = [col for col in data.columns if data[col].dropna().isin([0, 1, 2, 3, 4, 5, 6, 999]).all()]
    print("Categorical columns:", categorical_cols)

    for col in categorical_cols:
        data_imputed_df[col] = data_imputed_df[col].round().astype(int)
    # data_imputed_df
    #data_imputed_df.to_csv("./output.csv", index=False)
    return data_imputed_df

df_train = data_preprocess(data)
df_test = data_preprocess(data_test)
print(df_test.head())


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
seed = 825

def load_data(data):
    
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    return X, y


X, y = load_data(df_train)
x_test = df_test.iloc[:, 0:]
print(x_test.head())
# load_data('../Feature Selection/output_after_feature_selection.csv')

def feature_selection(X, y, count,x_test):
    # Train a Random Forest model
    model = RandomForestRegressor(random_state=seed)
    model.fit(X, y)

    # Get feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)

    # Select features with importance above a threshold
    selected_features = feature_importance[:count]["Feature"]
    print("Selected Features:\n", selected_features)
    X = X[selected_features]
    x_test = x_test[selected_features]
    return X, x_test


X,x_test = feature_selection(X, y, 28, x_test)
def zscoree(X, y_reg):
    # 假设 X 是特征，y_reg 是目标变量

    # 计算 Z-Score
    z_scores = zscore(X)

    # 计算所有特征的 Z-Score的绝对值
    z_scores_flat = np.abs(z_scores)

    # 绘制 Z-Score 分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(z_scores_flat, bins=30, kde=True)
    plt.title("Distribution of Z-Scores")
    plt.xlabel("Z-Score")
    plt.ylabel("Frequency")
    plt.axvline(x=3, color='r', linestyle='--', label="Outlier threshold = 3")
    plt.legend()
    plt.show()

    # 设置 Z-Score 阈值
    threshold = 3

    # 找到所有 Z-Score 超过阈值的样本（每个特征的 Z-Score 超过阈值的样本）
    outliers = (np.abs(z_scores) > threshold).any(axis=1)  # 检查任意一个特征的 Z-Score 是否超出阈值

    # 删除异常值
    X = X[~outliers]
    y_reg = y_reg[~outliers]
    # x_test = x_test[~outliers]
    
    # 输出去除的异常值个数
    print(f"Removed {np.sum(outliers)} outliers from the data.")
    print(X.shape, y_reg.shape)
    return X, y_reg


X, y = zscoree(X, y)


# def my_rf(X_train, y_train, x_test):
#     # Initialize the Random Forest Regressor
#     rf = RandomForestRegressor(n_estimators=100, random_state=seed)

#     # Train the model on the training data
#     rf.fit(X_train, y_train)

#     # Make predictions on the testing data
    
#     y_test_pred = rf.predict(x_test)
#     # Evaluate the model
    
#     return y_test_pred
# y_test_pred = my_rf(X, y, x_test)
# num = 1
# for i in y_test_pred:
#     print(f"datapoint {num} predicted value: ", i)
#     num += 1
# print(y_test_pred)


def split_data_into_test_and_train(X, y):
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data_into_test_and_train(X, y)

def my_rf(X_train, X_test, y_train, y_test, x_test):
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=seed)

    # Train the model on the training data
    rf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf.predict(X_test)
    y_test_pred = rf.predict(x_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    return y_test_pred



y_test_pred = my_rf(X_train, X_test, y_train, y_test, x_test)
num = 1
for i in y_test_pred:
    print(f"datapoint {num} predicted value: ", i)
    num += 1
# print(y_test_pred)
