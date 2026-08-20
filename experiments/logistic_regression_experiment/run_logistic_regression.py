import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from models.logistic_regression import LogisticRegression
from utils.data_preprocessing import (load_data, standardize, polynomial_features)
from utils.metrics import (mse, r2)



def run_experiment():
    '''
    ===============   Dataset   ===============
    '''
    # Dynamically build the path to the dataset using the root folder
    PATH = os.path.join(PROJECT_ROOT, "datasets", "Homes for Sale and Real Estate.csv")
    FEATURES = ["Sq.Ft", "Beds", "Bath"]
    TARGET = "Price"

    df, X, y = load_data(PATH, FEATURES, TARGET)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(6 * len(FEATURES), 5))

    for ax, feature in zip(np.atleast_1d(axes), FEATURES):
        ax.scatter(df[feature], df[TARGET])
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.grid(True)

    plt.tight_layout()
    save_figure("Features vs Target.png")
    plt.close()

    '''
    ===============   Training   ===============
    '''
    LEARNING_RATE = 0.001
    EPOCHS = 100000
    INIT = 'zeros'
    TOLERANCE = 1e-9
    STANDARDIZE = True

    if STANDARDIZE:
        X, mean, std = standardize(X)

    model_closed = LinearRegression(method='closed_form')
    model_gd = LinearRegression(method='gradient_descent')

    start = perf_counter()
    beta_closed, _, _ = model_closed.fit(
        X,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        init=INIT,
        tolerance=TOLERANCE
    )
    time_closed = perf_counter() - start

    start = perf_counter()
    beta_gd, losses, epoch = model_gd.fit(
        X,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        init=INIT,
        tolerance=TOLERANCE
    )
    time_gd = perf_counter() - start

    y_pred_closed = model_closed.predict(X)
    y_pred_gd = model_gd.predict(X)

    mse_closed = mse(y, y_pred_closed)
    mse_gd = mse(y, y_pred_gd)

    r2_closed = r2(y, y_pred_closed)
    r2_gd = r2(y, y_pred_gd)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.title("Gradient descent losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    save_figure("Gradient descent losses.png")
    plt.close()

    '''
    ===============   Evaluation   ===============
    '''
    print("======  Closed Form  ======")
    print("Beta: ")
    print(beta_closed)
    print(f"MSE: {mse_closed}")
    print(f"R2: {r2_closed}")
    print(f"Time: {time_closed}")

    print()

    print("======  Gradient Descent  ======")
    print("Beta: ")
    print(beta_gd)
    print(f"MSE: {mse_gd}")
    print(f"R2: {r2_gd}")
    print(f"Time: {time_gd}")
    print(f"number of epochs: {epoch}")

    print()

    print(f"MSE difference: {abs(mse_gd - mse_closed)}   (better: {'Gradient Descent' if mse_gd < mse_closed else 'Closed Form'})")
    print(f"R2 difference: {abs(r2_gd - r2_closed)}   (better: {'Gradient Descent' if r2_gd > r2_closed else 'Closed Form'})")
    print(f"Time difference: {abs(time_gd - time_closed)}   (better: {'Gradient Descent' if time_gd < time_closed else 'Closed Form'})")

    plt.subplots(1, 2, figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred_closed)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='perfect prediction')
    plt.title("Actual vs Predicted Price Using Closed Form")
    plt.xlabel("Actual")
    plt.ylabel("Predicted (Closed Form)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y, y_pred_gd)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='perfect prediction')
    plt.title("Actual vs Predicted Price Using Gradient Descent")
    plt.xlabel("Actual")
    plt.ylabel("Predicted (Gradient Descent)")
    plt.legend()
    plt.grid(True)

    save_figure("Actual vs Predicted Salary.png")
    plt.close()

    plt.subplots(1, 2, figsize=(16, 5))

    plt.subplot(1, 2, 1)
    residuals = y - y_pred_closed
    plt.scatter(y_pred_closed, residuals)
    plt.axhline(0, color='red')
    plt.title("Predicted vs Residuals")
    plt.xlabel("Predicted (Using Closed Form)")
    plt.ylabel("Residuals")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    residuals = y - y_pred_gd
    plt.scatter(y_pred_gd, residuals)
    plt.axhline(0, color='red')
    plt.title("Predicted vs Residuals")
    plt.xlabel("Predicted (Using Gradient Descent)")
    plt.ylabel("Residuals")
    plt.grid(True)

    save_figure("Predicted vs Residuals.png")
    plt.close()

    '''
    -------    Polynomial Regression (single feature)    -------
    '''
    degrees = [1, 2, 3, 5]

    plt.figure(figsize=(8, 5))

    x = X[:, 1].copy()
    idx = np.argsort(x)
    X_single = X[:, :2].copy()

    for degree in degrees:
        X_poly = polynomial_features(X_single, degree)
        beta, _, _ = model_closed.fit(X_poly, y)
        y_pred = model_closed.predict(X_poly)
        plt.plot(x[idx], y_pred[idx], linewidth=2, label=f"Degree {degree}")

    plt.scatter(x, y, color="black", s=15)
    plt.title("Polynomial Regression")
    plt.xlabel(FEATURES[0])
    plt.ylabel(TARGET)
    plt.legend()
    plt.grid(True)

    save_figure("polynomial (single feature)")
    plt.close()

    '''
    -------    Polynomial Degree vs MSE    -------
    '''
    degrees = range(1, 8)
    mses = []

    for degree in degrees:
        X_poly = polynomial_features(X, degree)
        beta, _, _ = model_closed.fit(X_poly, y)
        y_pred = model_closed.predict(X_poly)
        mses.append(mse(y, y_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, mses, marker="o")
    plt.title("Polynomial Degree vs MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.grid(True)

    save_figure("degree_vs_mse")
    plt.close()

    '''
    -------    Ridge: Lambda vs MSE    -------
    '''
    lambdas = [0, 0.01, 0.1, 1, 10, 100]
    mse_values = []

    for lam in lambdas:
        beta, _, _ = model_closed.fit(X, y, ridge_lambda=lam)
        y_pred = model_closed.predict(X)
        mse_values.append(mse(y, y_pred))

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas, mse_values, marker="o")
    plt.title("Ridge: Lambda vs MSE")
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.grid(True)

    save_figure("ridge_mse")
    plt.close()

    '''
    -------    Ridge Coefficient Shrinkage    -------
    '''
    lambdas = np.logspace(-4, 3, 100)
    coefficients = []

    for lam in lambdas:
        beta, _, _ = model_closed.fit(X, y, ridge_lambda=lam)
        coefficients.append(beta[1:].flatten())

    coefficients = np.array(coefficients)

    plt.figure(figsize=(8, 5))

    for i in range(coefficients.shape[1]):
        plt.semilogx(
            lambdas,
            coefficients[:, i],
            label=FEATURES[i]
        )

    plt.title("Ridge Coefficient Shrinkage")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.grid(True)

    save_figure("ridge_coefficients")
    plt.close()

def save_figure(filename: str):
    path = os.path.join(os.path.dirname(__file__), "figures", filename)
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    run_experiment()