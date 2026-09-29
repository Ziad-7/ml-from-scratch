import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.datasets import load_digits

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from models.neural_network import NN
from utils.data_preprocessing import train_test_split

def run_experiment():
    #-------------
    # X, y, model
    #-------------
    data = load_digits()
    X, y = data.data, data.target
    X = X / 16
    y = y.reshape(-1, 1)
    y_toNN = np.eye(10)[y.ravel()]
    nn = NN(layer_sizes=[64, 32, 10], output_activation='softmax')

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y_toNN)

    nn.fit(X_train, y_train, X_val, y_val, optimizer='adam', ridge_lambda=0.1)

    predictions = nn.predict(X_test)
    y_test_label = np.argmax(y_test, axis=1).reshape(-1, 1)
    print(np.mean(predictions == y_test_label) * 100)

    #-------------
    # Plots
    #-------------
    plt.figure()
    plt.plot(nn.train_losses, label='Training Loss')
    plt.plot(nn.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Digits Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(3, 4)
    axes = axes.ravel()
    for i in range(12):
        img = X_test[i].reshape(8, 8)
        prediction = predictions[i][0]
        actual = y_test_label[i][0]

        axes[i].imshow(img)

        color = 'green' if prediction == actual else 'red'

        axes[i].set_title(f"Pred: {prediction} | Actual: {actual}", color=color)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run_experiment()