"""
Neural Network Models for Weather Classification
Contains only the model classes without training dependencies
"""

import numpy as np


class NeuralNetwork(object):
    """Neural Network with 3 layers"""

    def __init__(self):
        self.input_unit = 7500
        self.hidden_units_1 = 512
        self.hidden_units_2 = 100
        self.output_class = 4
        # Xavier/He initialization for better training
        self.W1 = (np.random.randn(self.input_unit, self.hidden_units_1) * np.sqrt(2.0/self.input_unit)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_units_1, 1), dtype=np.float32)
        self.W2 = (np.random.randn(self.hidden_units_1, self.hidden_units_2) * np.sqrt(2.0/self.hidden_units_1)).astype(np.float32)
        self.b2 = np.zeros((self.hidden_units_2, 1), dtype=np.float32)
        self.W3 = (np.random.randn(self.hidden_units_2, self.output_class) * np.sqrt(2.0/self.hidden_units_2)).astype(np.float32)
        self.b3 = np.zeros((self.output_class, 1), dtype=np.float32)

    @staticmethod
    def softmax(Z):
        Z = Z - np.max(Z, axis=0, keepdims=True)
        e_Z = np.exp(Z, dtype=np.float64)
        A = e_Z / e_Z.sum(axis=0, keepdims=True)
        return A.astype(np.float32)

    @staticmethod
    def relu(Z): 
        return np.maximum(Z, 0)

    def feed_forward(self, X):
        self.Z1 = self.W1.T @ X + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2.T @ self.A1 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.W3.T @ self.A2 + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def back_propagation(self, X, Y, eta):
        N = X.shape[1]
        E3 = (self.A3 - Y)/N
        dW3 = self.A2 @ E3.T
        db3 = np.sum(E3, axis=1, keepdims=True)

        E2 = self.W3 @ E3
        E2[self.Z2 <= 0] = 0
        dW2 = self.A1 @ E2.T
        db2 = np.sum(E2, axis=1, keepdims=True)

        E1 = self.W2 @ E2
        E1[self.Z1 <= 0] = 0
        dW1 = X @ E1.T
        db1 = np.sum(E1, axis=1, keepdims=True)

        self.W1 -= eta*dW1; self.b1 -= eta*db1
        self.W2 -= eta*dW2; self.b2 -= eta*db2
        self.W3 -= eta*dW3; self.b3 -= eta*db3

    @staticmethod
    def cost(Y, Yhat):
        epsilon = 1e-7
        return float(-np.sum(Y*np.log(Yhat + epsilon))/Y.shape[1])

    def predict(self, X):
        y_hat = self.feed_forward(X)
        p = np.zeros_like(y_hat)
        p[np.argmax(y_hat, axis=0), np.arange(y_hat.shape[1])] = 1
        return p

    @staticmethod
    def score(predict, y):
        return round(float(np.mean(np.all(predict == y, axis=0))*100), 4)

    def accuracy(self, predict, y):
        """Alias for score() method"""
        return self.score(predict, y)


class NeuralNetworkV2(object):
    """Neural Network with 4 layers (deeper network)"""

    def __init__(self):
        self.input_unit = 7500
        self.hidden_units_1 = 1024
        self.hidden_units_2 = 512
        self.hidden_units_3 = 100
        self.output_class = 4
        # He initialization for ReLU networks
        self.W1 = (np.random.randn(self.input_unit, self.hidden_units_1) * np.sqrt(2.0/self.input_unit)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_units_1, 1), dtype=np.float32)
        self.W2 = (np.random.randn(self.hidden_units_1, self.hidden_units_2) * np.sqrt(2.0/self.hidden_units_1)).astype(np.float32)
        self.b2 = np.zeros((self.hidden_units_2, 1), dtype=np.float32)
        self.W3 = (np.random.randn(self.hidden_units_2, self.hidden_units_3) * np.sqrt(2.0/self.hidden_units_2)).astype(np.float32)
        self.b3 = np.zeros((self.hidden_units_3, 1), dtype=np.float32)
        self.W4 = (np.random.randn(self.hidden_units_3, self.output_class) * np.sqrt(2.0/self.hidden_units_3)).astype(np.float32)
        self.b4 = np.zeros((self.output_class, 1), dtype=np.float32)

    @staticmethod
    def softmax(Z):
        Z = Z - np.max(Z, axis=0, keepdims=True)
        e_Z = np.exp(Z, dtype=np.float64)
        A = e_Z / e_Z.sum(axis=0, keepdims=True)
        return A.astype(np.float32)

    @staticmethod
    def relu(Z): 
        return np.maximum(Z, 0)

    def feed_forward(self, X):
        self.Z1 = self.W1.T @ X + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2.T @ self.A1 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.W3.T @ self.A2 + self.b3
        self.A3 = self.relu(self.Z3)
        self.Z4 = self.W4.T @ self.A3 + self.b4
        self.A4 = self.softmax(self.Z4)
        return self.A4

    def back_propagation(self, X, Y, eta):
        N = X.shape[1]
        E4 = (self.A4 - Y)/N
        dW4 = self.A3 @ E4.T
        db4 = np.sum(E4, axis=1, keepdims=True)

        E3 = self.W4 @ E4
        E3[self.Z3 <= 0] = 0
        dW3 = self.A2 @ E3.T
        db3 = np.sum(E3, axis=1, keepdims=True)

        E2 = self.W3 @ E3
        E2[self.Z2 <= 0] = 0
        dW2 = self.A1 @ E2.T
        db2 = np.sum(E2, axis=1, keepdims=True)

        E1 = self.W2 @ E2
        E1[self.Z1 <= 0] = 0
        dW1 = X @ E1.T
        db1 = np.sum(E1, axis=1, keepdims=True)

        self.W1 -= eta*dW1; self.b1 -= eta*db1
        self.W2 -= eta*dW2; self.b2 -= eta*db2
        self.W3 -= eta*dW3; self.b3 -= eta*db3
        self.W4 -= eta*dW4; self.b4 -= eta*db4

    @staticmethod
    def cost(Y, Yhat):
        epsilon = 1e-7
        return float(-np.sum(Y*np.log(Yhat + epsilon))/Y.shape[1])

    def predict(self, X):
        y_hat = self.feed_forward(X)
        p = np.zeros_like(y_hat)
        p[np.argmax(y_hat, axis=0), np.arange(y_hat.shape[1])] = 1
        return p

    @staticmethod
    def score(predict, y):
        return round(float(np.mean(np.all(predict == y, axis=0))*100), 4)
    
    def accuracy(self, predict, y):
        """Alias for score() method"""
        return self.score(predict, y)
