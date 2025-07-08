import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize

class PrincipalComponentGaussianProcessModel:
    def __init__(self, n_components=9, input_dim=12, output_dim=28):
        self.n_components = n_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardization_mean = None
        self.standardization_scale = None
        self.K_eta_scores = None
        self.Phi_basis = None
        self.X_train = None
        self.X_train_std = None
        self.Y_train_std = None
        self.rho = np.ones((n_components, input_dim)) * 0.1
        self.lambda_w = np.ones(n_components) * 1.0
        self.noise_var = 1e-3

    def standardize_inputs(self, X, ranges):
        X = np.asarray(X, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        mins = ranges[:, 0]
        maxs = ranges[:, 1]
        ranges_width = maxs - mins
        ranges_width[ranges_width == 0] = 1.0
        X_standardized = (X - mins) / ranges_width
        return np.clip(X_standardized, 0.0, 1.0)

    def _standardize_output(self, Y):
        self.standardization_mean = np.mean(Y, axis=0)
        Y_centered = Y - self.standardization_mean
        self.standardization_scale = np.sqrt(np.mean(Y_centered ** 2))
        if self.standardization_scale == 0:
            self.standardization_scale = 1.0
        return Y_centered / self.standardization_scale

    def _unstandardize_output(self, Y_standardized):
        return Y_standardized * self.standardization_scale + self.standardization_mean

    def compute_principal_components(self, Y_standardized):
        y_tensor = tf.convert_to_tensor(Y_standardized, dtype=tf.float64)
        s, u, v = tf.linalg.svd(y_tensor, full_matrices=False)
        
        actual_components = min(self.n_components, s.shape[0], v.shape[0])
        
        K_eta_scores = (u @ tf.linalg.diag(s))[:, :actual_components].numpy()
        Phi_basis = tf.transpose(v[:actual_components, :]).numpy()
        return K_eta_scores, Phi_basis
    
    def _build_kernel_matrix(self, X1, X2=None, component_idx=None):
        if X2 is None:
            X2 = X1

        if component_idx is not None:
            variance = 1.0 / self.lambda_w[component_idx]
            rho = self.rho[component_idx, :]
            kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
            return kernel(X1, X2)
        else:
            K_blocks = []
            for i in range(self.n_components):
                variance = 1.0 / self.lambda_w[i]
                rho = self.rho[i, :]
                kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
                K_blocks.append(kernel(X1, X2))
            return scipy.linalg.block_diag(*K_blocks)

class GaussianKernel:
    def __init__(self, variance=1.0, rho=None, input_dim=12):
        self.variance = tf.Variable(variance, dtype=tf.float64)
        self.rho = tf.Variable(rho if rho is not None else tf.ones(input_dim, dtype=tf.float64), dtype=tf.float64)

    def __call__(self, X1, X2):
        X1 = tf.cast(X1, dtype=tf.float64)
        X2 = tf.cast(X2, dtype=tf.float64)

        rho_safe = tf.where(self.rho > 1e-6, self.rho, 1e-6)

        scaled_sq_diff = tf.square(X1[:, None, :] - X2[None, :, :]) / tf.square(rho_safe[None, None, :])
        exponent = -0.5 * tf.reduce_sum(scaled_sq_diff, axis=-1)
        return self.variance * tf.exp(exponent)

    def set_hyperparameters(self, variance, rho=None):
        self.variance.assign(variance)
        if rho is not None:
            self.rho.assign(rho)

def generate_test_data(n_train=50, n_test=20, input_dim=3, output_dim=5):
    np.random.seed(42)
    X_train = np.random.uniform(0, 1, (n_train, input_dim))
    ranges = [(0, 1)] * input_dim
    
    def true_func(x):
        return np.column_stack((
            np.sin(x[:, 0] * 2),
            x[:, 1] ** 2,
            x[:, 0] * x[:, 2],
            np.cos(x[:, 1] + x[:, 2]),
            np.exp(x[:, 0])
        ))
    
    Y_train = true_func(X_train) + np.random.normal(0, 0.05, (n_train, output_dim))
    X_test = np.random.uniform(0, 1, (n_test, input_dim))
    Y_test = true_func(X_test)
    
    return X_train, Y_train, X_test, Y_test, ranges, true_func