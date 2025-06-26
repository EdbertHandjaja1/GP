import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PrincipalComponentGaussianProcessModel:
    def __init__():
        pass
    
    def fit():
        pass

    def predict():
        pass

class Matern32Kernel:
    def __init__(self, variance, length_scale):
        """
        Matern 3/2 Kernel.
        Args:
            variance (float): Signal variance (σ²).
            length_scale (float): Length scale (ℓ).
        """
        self.variance = tf.Variable(variance, name="variance", dtype=tf.float64)
        self.length_scale = tf.Variable(length_scale, name="length_scale", dtype=tf.float64)

    def __call__(self, X1, X2):
        """
        Computes the covariance matrix between X1 and X2, with r scaled by ℓ
        Args:
            X1: Array of shape (n_samples_1, n_features).
            X2: Array of shape (n_samples_2, n_features).
        Returns:
            Covariance matrix of shape (n_samples_1, n_samples_2).
        """
        sq_dist = tf.reduce_sum(tf.square(X1[:, None] - X2), axis=-1)
        r = tf.sqrt(sq_dist) / self.length_scale
        return self._matern32(r)

    def _matern32(self, r):
        """
        Matern 3/2 kernel function.
        Source: TensorFlow Framework
        Args:
            r: the Euclidean distance between the input points
        Returns:
            Matern 3/2 Kernel with euclidian distance r
        """
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)