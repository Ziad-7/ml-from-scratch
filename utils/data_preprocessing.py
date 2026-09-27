import numpy as np
import pandas as pd

def load_data(path: str, features: list[str], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    '''
    Loads the data (features and target) from the path into numpy arrays
    '''
    df = pd.read_csv(path)

    df = df.dropna(axis=1, how='all').dropna().reset_index(drop=True)


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


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float=0.6,
        val_ratio: float=0.2,
        seed: int=42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    m = X.shape[0]
    indices = np.random.permutation(m)

    # shuffle
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_end = int(m * train_ratio)
    val_end = int(m * (train_ratio + val_ratio))

    X_train = X_shuffled[: train_end]
    y_train = y_shuffled[: train_end]

    X_val = X_shuffled[train_end: val_end]
    y_val = y_shuffled[train_end: val_end]

    X_test = X_shuffled[val_end:]
    y_test = y_shuffled[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
    
    