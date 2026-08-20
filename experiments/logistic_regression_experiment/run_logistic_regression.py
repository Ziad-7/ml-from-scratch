import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.logistic_regression import LogisticRegression
from utils.data_preprocessing import (load_data, standardize, polynomial_features)
from utils.metrics import (accuracy)



def run_experiment():
    '''
    ===============   Dataset   ===============
    '''
    PATH = os.path.join(PROJECT_ROOT, "datasets", "Breast Cancer Wisconsin.csv")
    all_cols = pd.read_csv(PATH, nrows=0).columns
    FEATURES = [col for col in all_cols if col not in ['diagnosis', 'id', 'Unnamed: 32']]
    TARGET = "diagnosis"

    df, X, y = load_data(PATH, FEATURES, TARGET)
    y = np.where(y == 'M', 1, 0)


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

    model = LogisticRegression()

    start = perf_counter()
    beta, losses, epoch = model.fit(
        X,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        init=INIT,
        tolerance=TOLERANCE
    )
    time = perf_counter() - start

    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.title("Gradient descent losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    save_figure("Gradient descent losses.png")
    plt.close()

    y = y.flatten()

    y_pred = model.predict(X)
    y_pred = (y_pred >= 0.5).astype(int)
    y_pred = y_pred.flatten()
    
    acc = accuracy(y, y_pred)
    TP = np.sum((y == 1) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))
    FP = np.sum((y == 0) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    confusion_matrix = np.array([[TP, FN],
                                 [FP, TN]])

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.matshow(confusion_matrix, cmap='Blues', alpha=0.7)

    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=confusion_matrix[i][j], va='center', ha='center', size='xx-large', weight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Malignant (1)', 'Predicted Benign (0)'])
    ax.set_yticklabels(['Actual Malignant (1)', 'Actual Benign (0)'])

    save_figure("Confusion Matrix.png")
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