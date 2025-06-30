import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PrincipalComponentGaussianProcessModel:
    def __init__(self, n_components=9, input_dim=12, output_dim=28):
        """
        Initializes the PCGP model
        Args:
            n_components (int): Number of principal components (q).
            input_dim (int): Dimension of input space (p).
            output_dim (int): Dimension of output space (n).
        """
        self.n_components = n_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardization_mean = None
        self.standardization_scale = None
        self.K_eta = None
        self.weights = None
        self.X_train = None  
        self.rho = None 
        self.lambda_w = None 
    
    def standardize_inputs(self, X, ranges):
        """
        Standardizes inputs to [0,1] range based on parameter bounds.
        Args:
            X (np.ndarray): Input design matrix (m x p).
            ranges (list): List of (min,max) tuples for each parameter.
        Returns:
            np.ndarray: Standardized inputs in [0,1]^p.
        """
        X = np.asarray(X, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        
        # standardize each feature to [0,1] using min-max scaling
        mins = ranges[:, 0]
        maxs = ranges[:, 1]
        ranges_width = maxs - mins
        
        # handle cases where range width is zero
        ranges_width[ranges_width == 0] = 1.0  
        
        X_standardized = (X - mins) / ranges_width
        
        # make sure they stay within [0, 1]
        X_standardized = np.clip(X_standardized, 0.0, 1.0)
        
        return X_standardized
    
    def _standardize_output(self, Y):
        """
        Standardizes simulations
        Args:
            Y (np.ndarry): Output matrix (m x n)
        Returns:
            np.ndarray: Standardized outputs
        """
        # center about mean
        self.standardization_mean = np.mean(Y, axis=0) 
        Y_centered = Y - self.standardization_mean

        # calculate the single scaling value based on the variance
        self.standardization_scale = np.sqrt(np.mean(Y_centered ** 2))
        if self.standardization_scale == 0:
            self.standardization_scale = 1.0

        # scale output so that its variance is 1
        Y_standardized = Y_centered / self.standardization_scale

        return Y_standardized

    def _unstandardize_output(self, Y_standardized):
        Y = Y_standardized * self.standardization_scale
        Y = Y + self.standardization_mean

        return Y

    def compute_principal_components(self, Y_standardized):
        """
        Performs SVD to obtain EOF basis vectors
        Args:
            Y (np.ndarray): Output matrix (m x n).
        Returns:
            np.ndarray: Basis matrix (n x q).
        """
        # svd decomp
        y_tensor = tf.convert_to_tensor(Y_standardized, dtype=tf.float64)
        s, u, v = tf.linalg.svd(tf.convert_to_tensor(y_tensor), full_matrices=False)

        # eof is first q columns of [U * D * sqrt(m)]
        m = Y_standardized.shape[0]  
        K_eta = (u @ tf.linalg.diag(s) / np.sqrt(m))[:, :self.n_components].numpy()

        # component loadings or weight
        weights = (np.sqrt(m) * v.T).numpy()

        return K_eta, weights

    def _build_kernel_matrix(self):
        """
        Constructs the block-diagonal kernel matrix K for all principal components.
        
        Returns:
            np.ndarray: Block-diagonal kernel matrix (m*q, m*q)
        """
        K_blocks = []
        
        for i in range(self.n_components):
            variance = 1.0 / self.lambda_w[i]
            rho = self.rho[i,:]              
            
            kernel = GaussianKernel(variance=variance, rho=rho)
            K_i = kernel(self.X_train, self.X_train)  
            K_blocks.append(K_i)
        
        K = scipy.linalg.block_diag(*K_blocks)
        return K

    def _negative_log_marginal_likelihood(self, observed_weights, weights, noise_precision):
        """
        Calculate the negative log marginal likelihood.

        Args:
            observed_weights 
            weights
            noise_precision

        Returns:
            float: Negative log marginal likelihood value
        """
        # w_hat - w
        w_hat = observed_weights.flatten()
        w = weights.flatten()
        residual = w_hat - w

        # Q = residual^T * (K^T * K) * residual
        K = self._build_kernel_matrix()
        Q = residual.T @ K @ residual

        pass

    def fit(self, X_train, Y_train):
        """
        Fits the PCGP model to training data using MCMC.
        Args:
            X_train (np.ndarray): Input design matrix (m x p).
            Y_train (np.ndarray): Output matrix (m x n).
            n_mcmc (int): Number of MCMC samples.
        Returns:
            self: Fitted model.
        """
        Y_standardized = self._standardize_output(Y_train)
        self.K_eta, self.weights = self.compute_principal_components(Y_standardized)
        pass


    def predict(self, X_new, return_std=False):
        """
        Predicts outputs for new inputs with uncertainty.
        Args:
            X_new (np.ndarray): New input points (k x p).
            return_std (bool): Whether to return std deviation.
        Returns:
            np.ndarray: Predicted means (k x n).
            (optional) np.ndarray: Predicted std deviations (k x n).
        """

class GaussianKernel:
    def __init__(self, variance=1.0, rho=None, input_dim=12):
        """
        The Gaussian covariance function.
        Args:
            variance: λ_wi^{-1} (precision-adjusted variance for PC i)
            rho: Vector of ρ_{i1}, ..., ρ_{ip} (per-dimension correlations)
            p: Input dimension
        """
        self.variance = tf.Variable(variance, name="variance", dtype=tf.float64)
        self.rho = tf.Variable(rho if rho is not None else tf.ones(input_dim, dtype=tf.float64))

    def __call__(self, X1, X2):
        """
        Computes the covariance matrix between X1 and X2
        Args:
            X1: Array of shape (n_samples_1, n_features).
            X2: Array of shape (n_samples_2, n_features).
        Returns:
            Covariance matrix of shape (n_samples_1, n_samples_2).
        """
        sq_diff = tf.square(X1[:, None] - X2)

        rho_factors = tf.pow(self.rho, 4.0 * sq_diff)

        R = tf.reduce_prod(rho_factors, axis=-1)

        return self.variance * R
    
    def set_hyperparameters(self, variance, rho=None):
        """
        Updates kernel hyperparameters.
        Args:
            variance (float): New signal variance.
            length_scale (float): New length scale.
        """
        self.variance.assign(variance)
        self.rho.assign(rho)
