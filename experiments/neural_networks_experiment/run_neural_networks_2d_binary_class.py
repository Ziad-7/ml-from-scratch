import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.neural_network import NN
from sklearn.datasets import make_moons, make_circles
from utils.data_preprocessing import train_test_split


def run_experiment(data='circles', n_samples=200, noise=0.15, factor=0.6):
    #-------------
    # X, y, model
    #-------------
    if data == 'moons':
         X, y = make_moons(n_samples=n_samples, noise=noise)
    elif data == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    else:
         print(f"Data {data} doesn't exist.")
         return

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    nn = NN(layer_sizes=[32, 16, 1], hidden_activation='leaky_relu')
    y = y.reshape(-1, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y)

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

    x0_min, x0_max = np.min(X[:, 0]) - 0.3, np.max(X[:, 0]) + 0.3
    x1_min, x1_max = np.min(X[:, 1]) - 0.3, np.max(X[:, 1]) + 0.3

    x0_line = np.linspace(x0_min, x0_max, 200)
    x1_line = np.linspace(x1_min, x1_max, 200)
    x0, x1 = np.meshgrid(x0_line, x1_line)
    grid_points = np.c_[x0.ravel(), x1.ravel()]

    def LivePlot2D(model, epoch):
        ax_boundary.cla()
        ax_loss.cla()
        a, _ = model.forward(grid_points)
        y_ = a[-1].reshape(x0.shape)
 
        ax_boundary.contourf(x0, x1, y_, levels=10, cmap='coolwarm', alpha=0.7)
        ax_boundary.contour(x0, x1, y_, levels=[0.5], colors='black', linewidths=2)
        ax_boundary.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', zorder=3)
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
            epochs=5000,
            optimizer='adam', 
            ridge_lambda=0.1,
            patience=500,
            callback=LivePlot2D,
            n_log=50)

    predicitons = nn.predict(X_test)
    acc = np.mean(predicitons == y_test) * 100
    
    LivePlot2D(nn, 0)
    ax_boundary.set_title(f"Best Model (Epoch {nn.best_epoch}). Test Accuracy = {acc}")
    ax_loss.axvline(x=nn.best_epoch, color='red', linestyle='--', alpha=0.8)

    if data == 'moons':
            save_figure("Prediction_Curve_2D_moons", "Prediction_Curves_2D_moons")
    elif data == 'circles':
        save_figure("Prediction_Curve_2D_circles", "Prediction_Curves_2D_circles")
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


def main():
    tests = 3
    data_list = ['moons', 'circles']
    for i in range(tests):
            data = np.random.choice(data_list)
            n_samples = np.random.randint(50, 301)
            noise = np.random.beta(1, 5)
            factor = np.random.beta(5.2, 5)
            run_experiment(data, n_samples, noise, factor)


if __name__ == "__main__":
     main()