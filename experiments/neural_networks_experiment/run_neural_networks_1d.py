import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.neural_network import NN


def run_experiment():
    #-------------
    # X, y, model
    #-------------
    nn = NN(hidden_activation='linear', layer_sizes=[3, 1])
    X = np.array([[1], [2], [3], [2], [50], [60] ,[55], [61]])
    y = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])

    #-------------
    # Plots
    #-------------
    plt.ion()
    fig, ax = plt.subplots()

    ax.scatter(X, y, edgecolors='green')
    ax.set_title("Live Neural Network Leanring")
    ax.set_xlabel("X")
    ax.set_ylabel("y")

    x_line = np.linspace(np.min(X) - 2, np.max(X) + 2, 500).reshape(-1, 1)
    graph,  = ax.plot(x_line, np.zeros_like(x_line))

    def LivePlot1D(model, epoch, loss):
        a, _ = model.forward(x_line)
        graph.set_ydata(a[-1])
        ax.set_title(f"epoch {epoch}: Loss = {loss}")
        plt.pause(0.02)

    #-------------
    # fit
    #-------------
    nn.fit(X, y, epochs=20000, callback=LivePlot1D)
    prediction = nn.predict([[5], [0], [30], [70]])
    print(prediction)

    save_figure("Prediction_Curve_1D")
    plt.ioff()
    plt.show()


def save_figure(filename: str):
    path = os.path.join(os.path.dirname(__file__), "figures", filename)
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    run_experiment()