import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm


def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    return X, y


def feature_selection(X, y, count, seed):
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
    return X


def zscoree(X, y_reg):
    # 假设 X 是特征，y_reg 是目标变量

    # 计算 Z-Score
    z_scores = zscore(X)

    # 计算所有特征的 Z-Score的绝对值
    # z_scores_flat = np.abs(z_scores)

    # # 绘制 Z-Score 分布图
    # plt.figure(figsize=(10, 6))
    # sns.histplot(z_scores_flat, bins=30, kde=True)
    # plt.title("Distribution of Z-Scores")
    # plt.xlabel("Z-Score")
    # plt.ylabel("Frequency")
    # plt.axvline(x=3, color='r', linestyle='--', label="Outlier threshold = 3")
    # plt.legend()
    # plt.show()

    # 设置 Z-Score 阈值
    threshold = 3

    # 找到所有 Z-Score 超过阈值的样本（每个特征的 Z-Score 超过阈值的样本）
    outliers = (np.abs(z_scores) > threshold).any(axis=1)  # 检查任意一个特征的 Z-Score 是否超出阈值

    # 删除异常值
    X = X[~outliers]
    y_reg = y_reg[~outliers]

    # 输出去除的异常值个数
    print(f"Removed {np.sum(outliers)} outliers from the data.")
    print(X.shape, y_reg.shape)
    return X, y_reg


def split_data_into_test_and_train(X, y):
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def linear_reg(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error (MAE):", mae)
    return mse, r2, mae


def my_SVM(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'kernel': ['rbf', 'poly'],  # Kernels to try
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'epsilon': [0.01, 0.1, 0.5, 1],  # Epsilon in the epsilon-SVR model
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]  # Kernel coefficient (for 'rbf', 'poly', 'sigmoid')
    }

    # Grid search
    grid_search = GridSearchCV(
        SVR(),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # Optimize for R-squared
        verbose=1,  # Show progress
        n_jobs=-1  # Use all available processors
    )
    grid_search.fit(X_train_scaled, y_train)

    # Best hyperparameters and their performance
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation R2 Score:", grid_search.best_score_)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Test Mean Squared Error (MSE):", mse)
    print("Test R-squared (R2):", r2)
    print('Test Mean Absolute Error (MAE):', mae)
    return mse, r2, mae


def my_rf(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    rf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    return mse, r2, mae


def my_mlp(X_train, X_test, y_train, y_test):
    model_mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='logistic', solver='adam', max_iter=2000,
                             alpha=0.0001, random_state=42)

    # 训练模型
    model_mlp.fit(X_train, y_train)

    # 预测
    y_pred = model_mlp.predict(X_test)

    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MLPRegressor Mean Squared Error: {mse}")
    print(f"MLPRegressor R^2 Score: {r2}")
    print(f"MLPRegressor MAE: {mae}")
    return mse, r2, mae


mse_list = dict()
mse_list['linear'] = []
mse_list['SVM'] = []
mse_list['RF'] = []
mse_list['MLP'] = []
r2_list = dict()
r2_list['linear'] = []
r2_list['SVM'] = []
r2_list['RF'] = []
r2_list['MLP'] = []
mae_list = dict()
mae_list['linear'] = []
mae_list['SVM'] = []
mae_list['RF'] = []
mae_list['MLP'] = []
for seed in tqdm(range(10, 20)):
    X, y = load_data('../output.csv')
    X = feature_selection(X, y, 30, seed)
    X, y = zscoree(X, y)
    X_train, X_test, y_train, y_test = split_data_into_test_and_train(X, y)

    mse, r2, mae = linear_reg(X_train, X_test, y_train, y_test)
    mse_list['linear'].append(mse)
    r2_list['linear'].append(r2)
    mae_list['linear'].append(mae)

    mse, r2, mae = my_SVM(X_train, X_test, y_train, y_test)
    mse_list['SVM'].append(mse)
    r2_list['SVM'].append(r2)
    mae_list['SVM'].append(mae)

    mse, r2, mae = my_rf(X_train, X_test, y_train, y_test)
    mse_list['RF'].append(mse)
    r2_list['RF'].append(r2)
    mae_list['RF'].append(mae)

    mse, r2, mae = my_mlp(X_train, X_test, y_train, y_test)
    mse_list['MLP'].append(mse)
    r2_list['MLP'].append(r2)
    mae_list['MLP'].append(mae)


def visualize(l, title):
    # Creating x-coordinates for each data point
    x = range(len(l))

    # Plotting the curve
    plt.plot(x, l, marker='o', linestyle='-', linewidth=2, markersize=6)

    # Adding labels and title
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Display the plot
    plt.grid(True)
    plt.show()


visualize(mse_list['linear'], 'Linear mse')
visualize(mse_list['SVM'], 'SVM mse')
visualize(mse_list['RF'], 'RF mse')
visualize(mse_list['MLP'], 'MLP mse')
visualize(r2_list['linear'], 'Linear r2')
visualize(r2_list['SVM'], 'SVM r2')
visualize(r2_list['RF'], 'RF r2')
visualize(r2_list['MLP'], 'MLP r2')
visualize(mae_list['linear'], 'Linear mae')
visualize(mae_list['SVM'], 'SVM mae')
visualize(mae_list['RF'], 'RF mae')
visualize(mae_list['MLP'], 'MLP mae')
