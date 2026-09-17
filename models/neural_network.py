import numpy as np

class NN:
    def __init__(self, activation: str = 'ReLU', layer_sizes: list = [32, 16, 4, 1]):
        self.W = []
        self.b = []
        self.layer_sizes = layer_sizes
        self.activation = activation


    def sequantial(self):
        pass


    def fit(self, X, y):
        n_features = X.shape[1]
        full_layer_sizes = [n_features] +  self.layer_sizes

        self.W = np.random.randn(n_features) * 0.01
        self.b = np.zeros(n_features)

        n_layers = len(full_layer_sizes)
        a = [X]
        for i in range(1, n_layers): 
            a.append(self.g(a[i-1] @ self.W[i] + self.b[i]))

        
        

    def g(self, Z):
        activations = {
            'linear' : lambda Z: Z,
            'ReLU' : lambda Z: max(0, Z),
            'leakyReLU' : lambda Z: max(0.01 * Z, Z),
            'sigmoid' : lambda Z: 1 / (1 + np.exp(-Z)),
            'softmax' : lambda Z: np.exp(Z) / np.sum(np.exp(Z)),
            'softplus' : lambda Z: np.log(1 + np.exp(Z)),
            'tanh' : lambda Z: np.tanh(Z),
            'perceptron' : lambda Z: 1 if Z > 0 else 0
        }
        return activations[self.activation](Z)

    
    def predict(self):
        pass

