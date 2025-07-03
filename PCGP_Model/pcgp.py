import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PrincipalComponentGaussianProcessModel:
    def __init__(self, n_components=9, input_dim=12, output_dim=28):
        self.n_components = n_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardization_mean = None
        self.standardization_scale = None
        self.K_eta = None  
        self.weights = None  
        self.X_train = None
        self.X_train_std = None
        self.Y_train_std = None
        self.rho = np.ones((n_components, input_dim))
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
        m = Y_standardized.shape[0]
        K_eta = (u @ tf.linalg.diag(s) / np.sqrt(m))[:, :self.n_components].numpy()
        weights = (np.sqrt(m) * tf.transpose(v)).numpy()
        return K_eta, weights

    def _build_kernel_matrix(self, X1, X2=None, component_idx=None):
        if X2 is None:
            X2 = X1
            
        if component_idx is not None:
            variance = 1.0 / self.lambda_w[component_idx]
            rho = self.rho[component_idx,:]
            kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
            return kernel(X1, X2)
        else:
            K_blocks = []
            for i in range(self.n_components):
                variance = 1.0 / self.lambda_w[i]
                rho = self.rho[i,:]
                kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
                K_blocks.append(kernel(X1, X2))
            return scipy.linalg.block_diag(*K_blocks)

    def _compute_kronecker_product(self, A, B):
        """Compute Kronecker product A ⊗ B as dense matrix"""
        A = tf.convert_to_tensor(A, dtype=tf.float64)
        B = tf.convert_to_tensor(B, dtype=tf.float64)
        return tf.einsum('ij,kl->ikjl', A, B).numpy().reshape(
            A.shape[0] * B.shape[0], 
            A.shape[1] * B.shape[1]
        )

    def _negative_log_marginal_likelihood(self, rho_flattened, lambda_w, noise_var):
        """
        Calculate the negative log marginal likelihood for PCGP model.
        
        Args:
            rho_flattened (np.ndarray): Flattened array of length scales (n_components * input_dim)
            lambda_w (np.ndarray): Array of precision parameters (n_components)
            noise_var (float): Noise variance parameter
            
        Returns:
            float: Negative log marginal likelihood value
        """
        # Reshape and update parameters
        self.rho = np.reshape(rho_flattened, (self.n_components, self.input_dim))
        self.lambda_w = lambda_w
        self.noise_var = noise_var
        
        m = self.X_train_std.shape[0]  
        q = self.n_components
        n = self.output_dim  

        # K(X,X) - size (N*q × N*q)
        K_XX = self._build_kernel_matrix(self.X_train_std)
        K_XX += tf.eye(m * q, dtype=tf.float64) * 1e-6

        # I_N ⊗ Φ - size (N*D × N*q)
        I_N = tf.eye(m, dtype=tf.float64)
        Phi = tf.constant(self.K_eta, dtype=tf.float64)
        I_N_kron_Phi = self._compute_kronecker_product(I_N, Phi)

        # Σ_YY = (I_N ⊗ Φ) K(X,X) (I_N ⊗ Φ)^T + δ² I_{N*D}
        term1 = tf.matmul(I_N_kron_Phi, tf.matmul(K_XX, tf.transpose(I_N_kron_Phi)))
        noise_term = tf.eye(m * n, dtype=tf.float64) * self.noise_var
        Sigma_YY = term1 + noise_term
        Sigma_YY += tf.eye(m * n, dtype=tf.float64) * 1e-6

        L_Sigma_YY = tf.linalg.cholesky(Sigma_YY)

        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Sigma_YY)))
        Y_flat = tf.constant(self.Y_train_std.flatten(), dtype=tf.float64)[:, None]
        alpha = tf.linalg.cholesky_solve(L_Sigma_YY, Y_flat)
        data_fit = tf.squeeze(tf.matmul(tf.transpose(Y_flat), alpha))
        constant = 0.5 * m * n * tf.math.log(2.0 * np.pi)

        return (0.5 * data_fit + 0.5 * log_det + constant).numpy()


    def fit(self, X_train, Y_train, ranges):
        """
        Fits the PCGP model to training data
        Args:
            X_train (np.ndarray): Input design matrix (m x p).
            Y_train (np.ndarray): Output matrix (m x n).
            ranges (list): List of (min,max) tuples for each parameter for standardizing inputs.
        Returns:
            self: Fitted model.
        """
        self.X_train = X_train
        self.X_train_std = self.standardize_inputs(X_train, ranges)
        self.Y_train_std = self._standardize_output(Y_train)
        self.K_eta, self.weights = self.compute_principal_components(self.Y_train_std)

        # Define the function for minimization
        def objective(params):
            rho = params[:self.n_components * self.input_dim]
            lambda_w = params[self.n_components * self.input_dim:-1]
            noise_var = params[-1]
            return self._negative_log_marginal_likelihood(rho, lambda_w, noise_var)

        initial_params = [
            self.rho.flatten(),
            self.lambda_w,
            np.array([self.noise_var])
        ]

        bounds = [(1e-5, None)] * (self.n_components * self.input_dim) + \
                [(1e-5, None)] * self.n_components + \
                [(1e-5, None)]  

        result = minimize(
                fun=objective,
                x0=initial_params,
                method='L-BFGS-B',
                bounds=bounds
        )
        
        opt_params = result.x
        self.rho = np.reshape(opt_params[:self.n_components * self.input_dim], 
                            (self.n_components, self.input_dim))
        self.lambda_w = opt_params[self.n_components * self.input_dim:-1]
        self.noise_var = opt_params[-1]

        return self

    def predict(self, X_new, ranges, return_std=False):
        """
        Makes predictions using the fitted PCGP model.
        
        Args:
            X_new (np.ndarray): New input locations to predict at (m_test x p).
            ranges (list): List of (min,max) tuples for each parameter for standardizing inputs.
            return_std (bool): If True, returns both mean and standard deviation of predictions.
            
        Returns:
            np.ndarray: Predicted mean values at X_new (m_test x n).
            (np.ndarray, np.ndarray): If return_std=True, returns tuple of (mean, std) where std has shape (m_test x n).
        """
        X_new_std = self.standardize_inputs(X_new, ranges)
        N = self.X_train_std.shape[0]
        N_test = X_new_std.shape[0]
        q = self.n_components
        n = self.output_dim

        # K(X,X) - size (N*q × N*q)
        K_XX = self._build_kernel_matrix(self.X_train_std)
        K_XX += tf.eye(N * q, dtype=tf.float64) * 1e-6

        # I_N ⊗ Φ - size (N*n × N*q)
        I_N = tf.eye(N, dtype=tf.float64)
        Phi = tf.constant(self.K_eta, dtype=tf.float64)
        I_N_kron_Phi = self._compute_kronecker_product(I_N, Phi)

        # Σ_YY = (I_N ⊗ Φ) K(X,X) (I_N ⊗ Φ)^T + δ² I_{N*n}
        Sigma_YY = tf.matmul(I_N_kron_Phi, tf.matmul(K_XX, tf.transpose(I_N_kron_Phi)))
        Sigma_YY += tf.eye(N * n, dtype=tf.float64) * self.noise_var
        Sigma_YY += tf.eye(N * n, dtype=tf.float64) * 1e-6
        L_Sigma_YY = tf.linalg.cholesky(Sigma_YY)

        # K(X*,X) - size (N_test*q × N*q)
        K_Xnew_X = self._build_kernel_matrix(X_new_std, self.X_train_std)

        # K(X*,X*) - size (N_test*q × N_test*q)
        K_Xnew_Xnew = self._build_kernel_matrix(X_new_std)

        # Y_cent flattened (N*n × 1)
        Y_cent_flat = tf.constant(self.Y_train_std.flatten(), dtype=tf.float64)[:, None]

        # μ_w = K(X*,X) (I_N ⊗ Φ)^T Σ_YY^{-1} Y_cent
        alpha = tf.linalg.cholesky_solve(L_Sigma_YY, Y_cent_flat)
        mu_w_flat = tf.matmul(K_Xnew_X, tf.matmul(tf.transpose(I_N_kron_Phi), alpha))
        mu_w = tf.reshape(mu_w_flat, (N_test, q)).numpy()

        # I_Ntest ⊗ Φ - size (N_test*n × N_test*q)
        I_Ntest = tf.eye(N_test, dtype=tf.float64)
        I_Ntest_kron_Phi = self._compute_kronecker_product(I_Ntest, Phi)

        # μ_f = Y_bar + (I_Ntest ⊗ Φ) μ_w
        mu_f_flat = tf.matmul(I_Ntest_kron_Phi, mu_w_flat)
        mu_f = tf.reshape(mu_f_flat, (N_test, n)).numpy()
        pred_mean = self._unstandardize_output(mu_f)

        if return_std:
            # Cov(w|Y) = K(X*,X*) - K(X*,X) (I_N ⊗ Φ)^T Σ_YY^{-1} (I_N ⊗ Φ) K(X,X*)
            term = tf.matmul(I_N_kron_Phi, K_Xnew_X)
            alpha_cov = tf.linalg.cholesky_solve(L_Sigma_YY, term)
            cov_w = K_Xnew_Xnew - tf.matmul(K_Xnew_X, tf.matmul(tf.transpose(I_N_kron_Phi), alpha_cov))

            # Cov(f|Y) = (I_Ntest ⊗ Φ) Cov(w|Y) (I_Ntest ⊗ Φ)^T
            cov_f = tf.matmul(I_Ntest_kron_Phi, tf.matmul(cov_w, tf.transpose(I_Ntest_kron_Phi)))

            # Cov(y|Y) = Cov(f|Y) + δ² I_{N_test*n}
            cov_y = cov_f + tf.eye(N_test * n, dtype=tf.float64) * self.noise_var

            pred_var = np.diag(cov_y.numpy()).reshape(N_test, n)
            pred_std = np.sqrt(pred_var) * self.standardization_scale
            return pred_mean, pred_std
        
        return pred_mean

class GaussianKernel:
    def __init__(self, variance=1.0, rho=None, input_dim=12):
        self.variance = tf.Variable(variance, dtype=tf.float64)
        self.rho = tf.Variable(rho if rho is not None else tf.ones(input_dim, dtype=tf.float64), dtype=tf.float64)

    def __call__(self, X1, X2):
        X1 = tf.cast(X1, dtype=tf.float64)
        X2 = tf.cast(X2, dtype=tf.float64)
        rho_safe = tf.where(self.rho > 1e-6, self.rho, 1e-6)
        scaled_sq_diff = tf.square(X1[:, None] - X2) / tf.square(rho_safe)
        exponent = -0.5 * tf.reduce_sum(scaled_sq_diff, axis=-1)
        return self.variance * tf.exp(exponent)

    def set_hyperparameters(self, variance, rho=None):
        """
        Updates kernel hyperparameters.
        Args:
            variance (float): New signal variance.
            rho (np.ndarray): New length scales (rho).
        """
        self.variance.assign(variance)
        if rho is not None:
            self.rho.assign(rho)
