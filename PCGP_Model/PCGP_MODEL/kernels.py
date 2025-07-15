import tensorflow as tf
import numpy as np

class GaussianKernel:
    def __init__(self, variance=1.0, rho=None, input_dim=12):
        self.variance = tf.Variable(variance, dtype=tf.float64)
        self.rho = tf.Variable(rho if rho is not None else tf.ones(input_dim, dtype=tf.float64), dtype=tf.float64)

    # def __call__(self, X1, X2):
    #     X1 = tf.cast(X1, dtype=tf.float64)
    #     X2 = tf.cast(X2, dtype=tf.float64)

    #     rho_safe = tf.where(self.rho > 1e-6, self.rho, 1e-6)

    #     scaled_sq_diff = tf.square(X1[:, None, :] - X2[None, :, :]) / tf.square(rho_safe[None, None, :])
    #     exponent = -0.5 * tf.reduce_sum(scaled_sq_diff, axis=-1)
    #     return self.variance * tf.exp(exponent)
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        sq_dist = np.sum((X1[:, None] - X2) ** 2 / self.rho ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sq_dist)

    def set_hyperparameters(self, variance, rho=None):
        self.variance.assign(variance)
        if rho is not None:
            self.rho.assign(rho)