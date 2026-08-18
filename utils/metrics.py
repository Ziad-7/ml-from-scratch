import numpy as np

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y - y_pred)**2)


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y)

    return 1 - (np.mean((y - y_pred)**2))/(np.mean((y - y_mean)**2))