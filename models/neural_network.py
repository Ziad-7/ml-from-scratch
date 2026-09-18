import numpy as np

class NN:
    def __init__(self,
        hidden_activation: str = 'ReLU',
        output_activation: str = 'sigmoid',
        layer_sizes: list = [32, 16, 4, 1]
        ):
        self.W = [None]
        self.b = [None]
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation


    def sequantial(self):
        pass


    def fit(self, X, y):
        n_features = X.shape[1]
        self.layer_sizes = [n_features] +  self.layer_sizes
        n_layers = len(self.layer_sizes)

        a = [X]
        Z = [None]
        for i in range(1, n_layers): # 1 -> 4
            self.W.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01)
            self.b.append(np.zeros((1, self.layer_sizes[i])))

            activation = self.output_activation if i == n_layers - 1 else self.hidden_activation
            Z.append(a[i-1] @ self.W[i] + self.b[i])
            a.append(self.g(activation, Z[i]))


    def g(self, activation, Z):
        activations = {
            'linear' : lambda Z: Z,
            'ReLU' : lambda Z: np.maximum(0, Z),
            'leakyReLU' : lambda Z: np.maximum(0.01 * Z, Z),
            'sigmoid' : lambda Z: 1 / (1 + np.exp(-Z)),
            'softmax' : lambda Z: np.exp(Z) / np.sum(np.exp(Z)),
            'softplus' : lambda Z: np.log(1 + np.exp(Z)),
            'tanh' : lambda Z: np.tanh(Z),
            'perceptron' : lambda Z: 1 if Z > 0 else 0
        }

        return activations[activation](Z)

    
    def predict(self):
        pass


def main():
    nn1 = NN()
    X = np.array([[1], [2], [3], [2], [50], [60] ,[55], [61]])
    y = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])
    nn1.fit(X, y)

if __name__ == "__main__":
    main()