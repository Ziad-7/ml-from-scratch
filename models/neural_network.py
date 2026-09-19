import numpy as np

class NN:
    def __init__(self,
        hidden_activation: str = 'relu',
        output_activation: str = 'sigmoid',
        layer_sizes: list = [32, 16, 4, 1]
        ):
        self.W = [None]
        self.b = [None]
        self.losses = []
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def _init_parameters(self, n_features: int) -> None:
        '''
        Initializes the random weights and biases of the network
        '''
        full_layers_sizes = [n_features] +  self.layer_sizes
        n_layers = len(full_layers_sizes)

        for i in range(1, n_layers):
            self.W.append(np.random.randn(full_layers_sizes[i-1], full_layers_sizes[i]) * 0.01)
            self.b.append(np.zeros((1, full_layers_sizes[i])))


    def forward(self, X) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Feeds X through all layers using W and b
        '''
        n_features = X.shape[1]
        full_layers_sizes = [n_features] +  self.layer_sizes

        n_layers = len(full_layers_sizes)
        a = [X]
        Z = [None]
        for i in range(1, n_layers):
            activation = self.output_activation if i == n_layers - 1 else self.hidden_activation
            Z.append(a[i-1] @ self.W[i] + self.b[i])
            a.append(self.g(activation, Z[i]))

        return a, Z


    def compute_cost(self, y_pred, y) -> float:
        return ((y - y_pred)**2)/2 # idk what's the loss, shouldn't it be log likelihood or smth

    def backward(self):
        pass


    def fit(self, X, y):
        n_features = X.shape[1]
        full_layers_sizes = [n_features] +  self.layer_sizes
        n_layers = len(full_layers_sizes)

        a = [X]
        Z = [None]
        for i in range(1, n_layers): # 1 -> 4
            self.W.append(np.random.randn(full_layers_sizes[i-1], full_layers_sizes[i]) * 0.01)
            self.b.append(np.zeros((1, full_layers_sizes[i])))

            activation = self.output_activation if i == n_layers - 1 else self.hidden_activation
            Z.append(a[i-1] @ self.W[i] + self.b[i])
            a.append(self.g(activation, Z[i]))


    def new_fit(self, X, y, epochs=10000, learning_rate=0.01) -> None:
        self._init_parameters(X.shape[1])

        for epoch in range(epochs):
            a, Z = self.forward(X)
            pass
        pass
        

    def g(self, activation, Z):
        activations = {
            'linear' : lambda Z: Z,
            'relu' : lambda Z: np.maximum(0, Z),
            'leaky_relu' : lambda Z: np.maximum(0.01 * Z, Z),
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