import pandas as pd
from matplotlib import pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    return X, y


from sklearn.ensemble import RandomForestRegressor


def feature_selection(X, y, count):
    # Train a Random Forest model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Get feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Select features with importance above a threshold
    selected_features = feature_importance[:count]["Feature"]
    X = X[selected_features]
    return X


def split_data_into_test_and_train(X, y):
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def linear_reg(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print("Mean Squared Error (MSE):", mse)
    # print("R-squared (R2):", r2)

    # print("Model Coefficients:", model.coef_)
    # print("Model Intercept:", model.intercept_)

    mae = mean_absolute_error(y_test, y_pred)
    # print("Mean Absolute Error (MAE):", mae)
    return mse, r2


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Feature scaling
def SVM(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],  # Kernels to try
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
    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Cross-Validation R2 Score:", grid_search.best_score_)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print("Test Mean Squared Error (MSE):", mse)
    # print("Test R-squared (R2):", r2)
    return mse, r2


X, y = load_data('../output.csv')
linear_reg_mse = []
linear_reg_r2 = []
SVM_mse = []
SVM_r2 = []
for i in range(1, len(X) + 1):
    X = feature_selection(X, y, i)
    X_train, X_test, y_train, y_test = split_data_into_test_and_train(X, y)
    l_mse, l_r2 = linear_reg(X_train, X_test, y_train, y_test)
    s_mse, s_r2 = SVM(X_train, X_test, y_train, y_test)
    print(
        f'with top {i} features selected, linear regression have mse {l_mse} r2 {l_r2}\n with  SVM mse {s_mse} r2 {s_r2}\n')
    linear_reg_mse.append(l_mse)
    linear_reg_r2.append(l_r2)
    SVM_mse.append(s_mse)
    SVM_r2.append(s_r2)


def visualize_list(arr):
    # Plot the list
    plt.figure(figsize=(8, 6))
    plt.plot(arr, marker='o', linestyle='-', label='Values')  # Line plot
    plt.scatter(range(len(arr)), arr, color='red', label='Points')  # Optional scatter plot

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Values vs Index')
    plt.grid()
    plt.legend()
    plt.show()


visualize_list(linear_reg_mse)
visualize_list(linear_reg_r2)
visualize_list(SVM_mse)
visualize_list(SVM_r2)
