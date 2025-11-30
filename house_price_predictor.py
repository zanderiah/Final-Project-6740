from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import numpy as np

class house_price_predictor():

    def __init__(self, data, y_axis_key, remove_outliers=False, z_threshold=3):
        self.original_data = data.copy()
        self.y_axis_key = y_axis_key
        
        # Apply outlier removal if enabled
        if remove_outliers:
            print("Removing outliers...")
            data = self.remove_outliers(data, y_axis_key, z_threshold)

        self.data = data
        self.y = self.data[y_axis_key].astype(int)
        self.x = self.data.drop(columns=[y_axis_key])

        # One-hot encode categorical variables
        self.X = pd.get_dummies(self.x, drop_first=True)

        # Models
        self.scaler = StandardScaler()
        self.linearRegressionModal = LinearRegression()

    # -------------------------------------------------
    # Outlier Removal Method
    # -------------------------------------------------
    def remove_outliers(self, df, col, threshold=3):
        """
        Removes outliers using Z-score thresholding on the target variable.
        """
        mean = df[col].mean()
        std = df[col].std()

        z_scores = (df[col] - mean) / std
        filtered_df = df[(np.abs(z_scores) < threshold)]

        print(f"Removed {len(df) - len(filtered_df)} outliers.")
        return filtered_df

    # -------------------------------------------------
    # Linear Regression
    # -------------------------------------------------
    def linear_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.linearRegressionModal.fit(X_train, y_train)
        predictions = self.linearRegressionModal.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print("R² Score:", r2)
        print("RMSE:", rmse)

    # -------------------------------------------------
    # LassoCV (auto alpha)
    # -------------------------------------------------
    def LassoCV(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

        model = LassoCV(cv=5, n_jobs=-1)
        model.fit(X_train_scaled, y_train_scaled)

        print("Optimal Alpha:", model.alpha_)
        print("R² (test):", model.score(X_test_scaled, y_test_scaled))
        print("Coefficients:", model.coef_)

    # -------------------------------------------------
    # Lasso Regression (manual alpha)
    # -------------------------------------------------
    def LassoRegression(self, alpha=0.001):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train_scaled, y_train)

        y_pred = lasso_model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)

        n = len(y_test)
        p = np.sum(lasso_model.coef_ != 0)
        adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

        print("R²:", r2)
        print("Adjusted R²:", adj_r2)
        print("Coefficients:", lasso_model.coef_)

    # -------------------------------------------------
    # Error Calculation
    # -------------------------------------------------
    def calculate_errors(self):
        predictions = self.linearRegressionModal.predict(self.X)

        mae = mean_absolute_error(self.y, predictions)
        mse = mean_squared_error(self.y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y, predictions)

        print("Dataset Errors:")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
