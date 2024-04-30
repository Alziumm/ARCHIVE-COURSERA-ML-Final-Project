import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(filepath):

    return pd.read_csv(filepath)

def preprocess_data(df):

    features = df[['AT', 'AP', 'RH', 'V']]
    target = df['PE']

    return features, target

def split_data(features, target):

    return train_test_split(features, target, test_size=0.3)

def train_evaluate_model(X_train, X_test, y_train, y_test):

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    lr_mse = -np.mean(lr_scores)

    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rf_mse = -np.mean(rf_scores)

    print("----------------------------")
    print(f"Linear model return a MSE of {lr_mse} on the test set.")
    print(f"Random Forest model return a MSE of {rf_mse} on the test set.")

    best_model = rf_model if rf_mse < lr_mse else lr_model
    model_type = "Random Forest" if rf_mse < lr_mse else "Linear Regression"

    final_mse = mean_squared_error(y_test, best_model.predict(X_test))

    return model_type, final_mse

def main(filepath):

    df = load_data(filepath)

    features, target = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(features, target)

    best_model_type, test_mse = train_evaluate_model(X_train, X_test, y_train, y_test)

    print("----------------------------")
    print(f"The best model is {best_model_type} with a MSE of {test_mse} on the test set.")
    print("----------------------------")

main('CCPP_data.csv')