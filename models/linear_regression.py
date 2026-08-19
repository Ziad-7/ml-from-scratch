import numpy as np

class LinearRegression:
    def __init__(self, method='gradient_descent'):
        self.beta = None
        self.method = method
        self.losses = []
        self.epochs_ran = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100000,
        init: str = 'zeros',
        tolerance: float = 1e-6,
        ridge_lambda: float = 0
        ) -> tuple[np.ndarray, list[float], int]:
        '''
        Fits the regession line

        Parameters
        ----------
        X : input features
        y : given output
        learning_rate : learning rate
        epochs : number of iterationsz
        init : inital values for features' coefficient
        tolerance : stopping error criterion
        ridge_lambda : lambda for regularization

        Returns
        -------
        beta : features' coefficient
        losses : loss history
        epochs_ran : number of iterations done
        '''

        m, n = X.shape

        if init == 'zeros':
            self.beta = np.zeros((n, 1))

        elif init == 'random':
            self.beta = np.random.randn(n, 1)
        
        elif init == 'large_random':
            self.beta = np.random.randn(n, 1) * 100
            
        else:
            raise ValueError('Unknown Initilization')


        if self.method == 'closed_form':
            I = np.eye(n)
            I[0, 0] = 0

            A = X.T @ X + ridge_lambda * I
            B = X.T @ y

            self.beta = np.linalg.solve(A, B)

        elif self.method == 'gradient_descent':
            prev_loss = np.inf

            for epoch in range(epochs):
                y_pred = self.predict(X)
                
                ridge_beta = self.beta.copy()
                ridge_beta[0] = 0

                loss = (
                ((y_pred - y).T @ (y_pred - y)) / (2 * m)
                +
                (ridge_lambda * ridge_beta.T @ ridge_beta) / (2 * m)
                ).item()
                
                self.losses.append(loss)

                gradient = (X.T @ (y_pred - y)) / m + (ridge_lambda * ridge_beta) / m
                
                # stopping criteria

                norm = np.linalg.norm(gradient)
                delta = abs(prev_loss - loss)

                if norm < tolerance or delta < tolerance:
                    break

                self.beta = self.beta - learning_rate * gradient
                prev_loss = loss
                self.epochs_ran = epoch
        else:
            raise ValueError('Unknown Method')
            
        return self.beta, self.losses, self.epochs_ran + 1


    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta
