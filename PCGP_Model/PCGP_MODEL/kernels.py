import tensorflow as tf
import numpy as np

class GaussianKernel:
    def __init__(self, variance=1.0, rho=None, input_dim=12):
        self.variance = tf.Variable(variance, dtype=tf.float64)
        self.rho = tf.Variable(rho if rho is not None else tf.ones(input_dim, dtype=tf.float64), dtype=tf.float64)
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        sq_dist = np.sum((X1[:, None] - X2) ** 2 / self.rho ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sq_dist)

    def set_hyperparameters(self, variance, rho=None):
        self.variance.assign(variance)
        if rho is not None:
            self.rho.assign(rho)

            