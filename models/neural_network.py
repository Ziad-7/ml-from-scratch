import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self,
        hidden_activation: str = 'relu',
        output_activation: str = 'sigmoid',
        layer_sizes: list = [32, 16, 4, 1]
        ):
        self.W = [None]
        self.b = [None]
        self.train_losses = []
        self.val_losses = []
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def _init_parameters(self, n_features: int) -> None:
        '''
        Initializes the random weights and biases of the network
        '''
        self.W = [None]
        self.b = [None]
        self.train_losses = []
        self.val_losses = []
        full_layers_sizes = [n_features] +  self.layer_sizes
        n_layers = len(full_layers_sizes)

        for i in range(1, n_layers):
            self.W.append(np.random.randn(full_layers_sizes[i-1], full_layers_sizes[i]) * (2 / full_layers_sizes[i-1]))
            self.b.append(np.zeros((1, full_layers_sizes[i])))


    def forward(self, X) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Feeds X through all layers using W and b
        returns a and Z lists
        '''
        n_features = len(next(iter(X)))
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
        eps = 1e-15
        return np.mean(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))

    def backprop(self, a, Z, y):
        m = a[0].shape[0]
        n_layers = len(self.W)
        
        dJdW = [None] * n_layers
        dJdb = [None] * n_layers

        # dJ/dW = dJ/da * da/dZ * dZ/dW
        # dJ/db = dJ/da * da/dZ * dZ/db

        # dJ/da (final layer) = (a - y) / (a * (1 - a))
        # da/dZ = a * (1 - a)
        # dZ/dW = a_prev
        # dZ/db = 1

        dJdZ = a[-1] - y
        dJdW[-1] = (a[-2].T @ dJdZ) / m
        dJdb[-1] = np.sum(dJdZ, axis=0, keepdims=True) / m

        for i in range(n_layers - 2, 0, -1):
            # dJ/da (hidden layers) = dJ/dZ_next * dZ_next/da
            #   dJ/dZ_next = calculated
            #   dZ_next/da = W_next
            # da/dZ = 1 if Z > 0 else 0
            # dZ/dW = a_prev
            # dZ/db = 1

            dJda = dJdZ @ self.W[i+1].T
            dadZ = (Z[i] > 0).astype(float)
            dJdZ = dJda * dadZ

            dJdW[i] = (a[i-1].T @ dJdZ) / m
            dJdb[i] = np.sum(dJdZ, axis=0, keepdims=True) / m

        return dJdW, dJdb


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray=None,
            y_val: np.ndarray=None,
            learning_rate: float=0.01,
            epochs: int=10000,
            optimizer: str='gd',
            tolerance: float = 1e-12,
            patience: int=100,
            callback=None
        ) -> None:
        self._init_parameters(len(next(iter(X))))

        M_w = [np.zeros_like(w) if w is not None else 0 for w in self.W]
        M_b = [np.zeros_like(b) if b is not None else 0 for b in self.b]
        V_w = [np.zeros_like(w) if w is not None else 0 for w in self.W]
        V_b = [np.zeros_like(b) if b is not None else 0 for b in self.b]

        beta1 = 0.9
        beta2 = 0.99
        t = 0

        eps = 1e-8
        prev_loss = 0
        patience_counter = 0

        if X_val is not None and y_val is not None:
            best_val_loss = 1e9
            best_W = None
            best_b = None
        else:
            best_val_loss = None
            val_loss = None

        for epoch in range(epochs):
            t = epoch + 1

            a, Z = self.forward(X)
            train_loss = self.compute_cost(a[-1], y)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                a_, Z_ = self.forward(X_val)
                val_loss = self.compute_cost(a_[-1], y_val)
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W = [w.copy() if w is not None else None for w in self.W]
                    best_b = [b.copy() if b is not None else None for b in self.b]
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if abs(train_loss - prev_loss) < tolerance or patience_counter > patience:
                print(f"Early stopping at epoch={epoch}")
                break

            prev_loss = train_loss

            dW, db = self.backprop(a, Z, y)
            for i in range(1, len(self.W)):
                if optimizer == 'gd':
                    self.W[i] = self.W[i] - learning_rate * dW[i]
                    self.b[i] = self.b[i] - learning_rate * db[i]
                elif optimizer == 'adam':
                    M_w[i] = beta1 * M_w[i] + (1 - beta1) * dW[i]
                    M_b[i] = beta1 * M_b[i] + (1 - beta1) * db[i]

                    V_w[i] = beta2 * V_w[i] + (1 - beta2) * dW[i]**2
                    V_b[i] = beta2 * V_b[i] + (1 - beta2) * db[i]**2

                    M_w_hat = M_w[i] / (1 - beta1 ** t)
                    M_b_hat = M_b[i] / (1 - beta1 ** t)

                    V_w_hat = V_w[i] / (1 - beta2 ** t)
                    V_b_hat = V_b[i] / (1 - beta2 ** t)

                    self.W[i] = self.W[i] - learning_rate * M_w_hat / (np.sqrt(V_w_hat) + eps)
                    self.b[i] = self.b[i] - learning_rate * M_b_hat / (np.sqrt(V_b_hat) + eps)
            
            if epoch % 50 == 0:
                print(f"epoch {epoch}: Train Loss = {train_loss} - Validation Loss = {val_loss} - Best Validation Loss = {best_val_loss}")
                if callback:
                    callback(self, epoch)

        if X_val is not None and y_val is not None:
            self.W = best_W
            self.b = best_b
        

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


    def predict_prob(self, X):
        a, _ = self.forward(X)
        return a[-1]


    def predict(self, X):
        a, Z = self.forward(X)
        return (a[-1] > 0.5).astype(int)