import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.neural_network import NN
from sklearn.datasets import make_blobs
from utils.data_preprocessing import train_test_split


def run_experiment(n_classes=2):
    #-------------
    # X, y, model
    #-------------
    X, y = make_blobs(n_samples=200, centers=n_classes)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    y = y.reshape(-1, 1)
    y_toNN = np.eye(n_classes)[y.ravel()]
    nn = NN(layer_sizes=[32, 16, 8, 4, n_classes], output_activation='softmax')

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y_toNN)

    #-------------
    # Plots
    #-------------
    plt.ion()
    fig, (ax_boundary, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    ax_boundary.scatter(X[:, 0], X[:, 1], c=y)
    ax_boundary.set_title("Live Neural Network Leanring")
    ax_boundary.set_xlabel("X")
    ax_boundary.set_ylabel("y")
    ax_boundary.set_xticks([])
    ax_boundary.set_yticks([])

    x0_min, x0_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x1_min, x1_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

    x0_line = np.linspace(x0_min, x0_max, 200)
    x1_line = np.linspace(x1_min, x1_max, 200)
    x0, x1 = np.meshgrid(x0_line, x1_line)
    grid_points = np.c_[x0.ravel(), x1.ravel()]

    def LivePlot2D(model, epoch):
        ax_boundary.cla()
        ax_loss.cla()
        y_ = model.predict(grid_points).reshape(x0.shape)

        levelsf = [-0.5 + i for i in range(n_classes + 1)]
        levels = [0.5 + i for i in range(n_classes - 1)]
        ax_boundary.contourf(x0, x1, y_, levels=levelsf, cmap='Set1', alpha=0.7)
        ax_boundary.contour(x0, x1, y_, levels=levels, colors='black', linewidths=2)
        ax_boundary.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='Set1', zorder=3)
        ax_boundary.set_title(f"Epoch {epoch}")

        ax_loss.plot(nn.train_losses, label="Training Loss")
        ax_loss.plot(nn.val_losses, label="Validation Loss")
        ax_loss.set_title("Loss Curves")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)

    nn.fit(
        X_train, y_train, X_val, y_val,
        epochs=10000,
        optimizer='adam', 
        ridge_lambda=0.1,
        patience=500,
        callback=LivePlot2D,
        n_log=10)
    
    predicitons = nn.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1).reshape(-1, 1)
    acc = np.mean(predicitons == y_test_labels) * 100

    LivePlot2D(nn, 0)
    ax_boundary.set_title(f"Best Model (Epoch {nn.best_epoch}). Test Accuracy = {acc}")
    ax_loss.axvline(x=nn.best_epoch, color='red', linestyle='--', alpha=0.8)
    # save_figure("Prediction_Curve_2D_moons")
    # save_figure("Prediction_Curve_2D_circles")
    save_figure("Prediction_Curve_2D_blobs", "Prediction_Curves_2D_blobs")
    plt.pause(2)
    plt.close(fig)
    
    

def save_figure(filename: str, foldername: str):
    folder = os.path.join(os.path.dirname(__file__), "figures", foldername)
    n_files = len(os.listdir(folder)) + 1
    filename = f"{filename}_{n_files}.png"
    path = os.path.join(folder, filename)
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    tests = 3
    n_classes = [np.random.randint(2, 6) for i in range(tests)]
    for i in range(tests):
        run_experiment(n_classes[i])
