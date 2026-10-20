import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    data_path = "Titanic-Dataset.csv"
    df = pd.read_csv(data_path)

    df_reg = df[['Fare', 'Age', 'Pclass', 'SibSp', 'Parch']].dropna()

    X = df_reg[['Age', 'Pclass', 'SibSp', 'Parch']].values
    y = df_reg['Fare'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Regression evaluation:")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2: {r2:.3f}")

    coef_names = ['Age', 'Pclass', 'SibSp', 'Parch']
    print('\nModel coefficients:')
    for name, coef in zip(coef_names, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title('Actual vs Predicted Fare')
    plt.tight_layout()
    plt.show()

    print('\nInterpretation:')
    print('- Positive coefficient means that feature increases Fare when all else equal.')
    print('- R2 near 0 indicates linear model explains little variance in Fare (common for this dataset).')


if __name__ == '__main__':
    main()
