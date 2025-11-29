from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

class house_price_predictor():

    def __init__(self, data, y_axis_key):
        self.data = data
        self.y = self.data[y_axis_key].astype(int)
        self.x = self.data.drop(columns=[y_axis_key])
        self.X = pd.get_dummies(self.x, drop_first=True)
        self.scaler = StandardScaler()

    def linear_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("R² Score:", r2)
        print("RMSE:", rmse)

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

        model = LassoCV(cv=5, n_jobs=-1) # n_jobs=-1 uses all available CPU cores
        model.fit(X_train_scaled, y_train_scaled)
        optimal_alpha = model.alpha_
        print(f"The optimal alpha value is: {optimal_alpha}")
        score = model.score(X_test_scaled, y_test_scaled)
        print(f"R-squared score on test set: {score}")
        print(f"Coefficients: {model.coef_}")

    def LassoRegression(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()
        
        lasso_model = Lasso(alpha=0.001)
        lasso_model.fit(X_train_scaled, y_train)
        
        y_pred = lasso_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        
        r2 = r2_score(y_test,y_pred)
        print(f"Coefficients: {lasso_model.coef_}")
        
        n = len(y_test)
        p = np.sum(lasso_model.coef_ != 0)
        
        adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
        print(r2)
        print(adj_r2)
    