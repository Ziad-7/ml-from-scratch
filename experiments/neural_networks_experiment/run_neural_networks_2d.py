import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.neural_network import NN
from sklearn.datasets import make_moons
from utils.data_preprocessing import train_test_split


def run_experiment():
    #-------------
    # X, y, model
    #-------------
    nn = NN(layer_sizes=[16, 8, 4, 1], hidden_activation='relu')
    X, y = make_moons(n_samples=200, noise=0.2)
    y = y.reshape(-1, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y)

    #-------------
    # plot
    #-------------
    plt.ion()
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title("Live Neural Network Leanring")
    ax.set_xlabel("X")
    ax.set_ylabel("y")

    x0_min, x0_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x1_min, x1_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

    x0_line = np.linspace(x0_min, x0_max, 200)
    x1_line = np.linspace(x1_min, x1_max, 200)
    x0, x1 = np.meshgrid(x0_line, x1_line)
    grid_points = np.c_[x0.ravel(), x1.ravel()]

    def LivePlot2D(model, epoch):
        ax.cla()
        a, _ = model.forward(grid_points)
        y_ = a[-1].reshape(x0.shape)
 
        ax.contourf(x0, x1, y_, levels=10, cmap='coolwarm', alpha=0.7)
        ax.contour(x0, x1, y_, levels=[0.5], colors='black', linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm')
        if isinstance(epoch, int):
            ax.set_title(f"epoch {epoch}")
        elif isinstance(epoch, str):
            ax.set_title(epoch)
        plt.pause(0.02)

    nn.fit(X_train, y_train, X_val, y_val, epochs=10000, optimizer='adam', callback=LivePlot2D)
    plt.pause(1)
    LivePlot2D(nn, "Best Model")
    save_figure("Prediction_Curve_2D")
    plt.ioff()
    plt.show()

    fig2 = plt.plot(nn.train_losses, label="Training Loss")
    plt.plot(nn.val_losses, label="Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    predicitons = nn.predict(X_test)
    print(np.mean(predicitons == y_test) * 100)
    

def save_figure(filename: str):
    path = os.path.join(os.path.dirname(__file__), "figures", filename)
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    run_experiment()