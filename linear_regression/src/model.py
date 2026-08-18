import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def load_data(path: str, features: list[str], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    '''
    Loads the data (features and target) from the path into numpy arrays
    '''
    df = pd.read_csv(path)

    df = df.dropna().reset_index(drop=True)

    assert len(features) > 0
    assert target in df.columns

    for feature in features:
        assert feature in df.columns

    X = df[features].to_numpy()
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    y = df[target].to_numpy()
    y = y.reshape(-1, 1)

    return df, X, y


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Standarizes the features using z-score normalization
    '''

    X_scaled = X.copy()

    mu = X[:, 1:].mean(axis=0)
    sigma = X[:, 1:].std(axis=0)

    if np.any(sigma == 0):
        raise ValueError("A feature has zero variance.")

    X_scaled[:, 1:] = (X[:, 1:] - mu) / sigma

    return X_scaled, mu, sigma


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    '''
    Expands the features matrix with polynomial terms
    '''

    X_poly = X[:, :1].copy()

    features = X[:, 1:].copy()

    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, features ** d))

    return X_poly


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y - y_pred)**2)


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y)

    return 1 - (np.mean((y - y_pred)**2))/(np.mean((y - y_mean)**2))


def fit_closed_form(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 0) -> np.ndarray:
    '''
    Fits the regession line using the closed formula

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    ridge_lambda : float

    Returns
    -------
    beta : np.ndarray
    '''

    I = np.eye(X.shape[1])
    I[0, 0] = 0

    A = X.T @ X + ridge_lambda * I
    B = X.T @ y

    beta = np.linalg.solve(A, B)

    return beta


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100000,
    init: str = 'zeros',
    tolerance: float = 1e-6,
    ridge_lambda: float = 0
    ) -> tuple[np.ndarray, list[float], int]:
    '''
    Fits the regession line using Gradient Descent

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    learning_rate : float
    epochs : int
    init : str
    tolerance : float
    ridge_lambda : float

    Returns
    -------
    beta : np.ndarray
    losses : list[float]
    epoch : float
    '''

    m, n = X.shape
    
    if init == 'zeros':
        beta = np.zeros((n, 1))

    elif init == 'random':
        beta = np.random.randn(n, 1)
    
    elif init == 'large_random':
        beta = np.random.randn(n, 1) * 100
        
    else:
        raise ValueError('Unknown Initilization')

    losses = []
    prev_loss = np.inf

    for epoch in range(epochs):
        y_pred = predict(X, beta)
        
        reg_beta = beta.copy()
        reg_beta[0] = 0

        loss = (
        ((y_pred - y).T @ (y_pred - y)) / (2 * m)
        +
        (ridge_lambda * reg_beta.T @ reg_beta) / (2 * m)
        ).item()
        
        losses.append(loss)

        gradient = (
        ((X.T) @ (y_pred - y)) / m
        +
        (ridge_lambda / m) * reg_beta
        )
        
        # stopping criteria

        norm = np.linalg.norm(gradient)
        delta = abs(prev_loss - loss)

        if norm < tolerance or delta < tolerance:
            break

        beta = beta - learning_rate * gradient
        prev_loss = loss

    return beta, losses, epoch + 1


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


def save_figure(filename: str):
    plt.savefig(
        f"../figures/{filename}",
        dpi=300,
        bbox_inches="tight"
    )