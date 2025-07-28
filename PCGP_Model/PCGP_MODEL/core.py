import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PCGP_MODEL.kernels import GaussianKernel

class PrincipalComponentGaussianProcessModel:
    """
    A class to implement a Principal Component Gaussian Process (PCGP) model.
    This model combines Principal Component Analysis (PCA) with Gaussian Processes (GPs)
    to model high-dimensional outputs. PCA is used to reduce the dimensionality of the
    output space, and then individual GPs are trained on the principal component weights.
    
    Optimized version with matrix caching to prevent recomputation of expensive operations.
    """
    def __init__(self, n_components=9, input_dim=12, output_dim=28):
        """
        Sets up the initial parameters of the PCGP model,

        Arguments:
            n_components (int): The number of principal components to retain. 
            input_dim (int): The dimensionality of the input data (X). 
            output_dim (int): The dimensionality of the output data (Y). 

        Output:
            None. 
        """
        self.n_components = n_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardization_mean = None
        self.standardization_scale = None
        self.weights = None
        self.phi_basis = None
        self.X_train = None
        self.Y_train = None
        self.X_train_std = None
        self.Y_train_std = None
        self.rho = np.ones((n_components, input_dim)) * 0.1
        self.lambda_w = np.ones(n_components) * 1.0
        self.noise_var = 1e-3
        
        # store matrices
        self._stored_kernels = [None] * n_components  
        self._stored_cholesky = [None] * n_components 
        self._stored_sigma = [None] * n_components    
        self._last_hyperparams = None  
        
    def _hyperparams_changed(self, rho_flattened, lambda_w, noise_var):
        """Check if hyperparameters have changed since last computation."""
        current_params = (tuple(rho_flattened), tuple(lambda_w), noise_var)
        if self._last_hyperparams is None or self._last_hyperparams != current_params:
            self._last_hyperparams = current_params
            return True
        return False
    
    def _compute_and_store_matrices(self):
        """Compute and store all kernel matrices and Cholesky decompositions."""
        n = self.X_train_std.shape[0]
        X_train_tf = tf.convert_to_tensor(self.X_train_std, dtype=tf.float64)
        
        for k in range(self.n_components):
            variance = 1.0 / self.lambda_w[k]
            rho = self.rho[k, :]
            kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
            K_k = kernel(X_train_tf, X_train_tf)
            self._stored_kernels[k] = K_k
            
            Sigma_k = K_k + self.noise_var * tf.eye(n, dtype=tf.float64)
            self._stored_sigma[k] = Sigma_k
            
            L_k = tf.linalg.cholesky(Sigma_k)
            
            self._stored_cholesky[k] = L_k

    def standardize_inputs(self, X, ranges):
        """
        Standardizes input data X to the range [0, 1] based on provided min/max ranges.

        Arguments:
            X (np.ndarray): The input data to be standardized. Shape (n_samples, input_dim).
            ranges (np.ndarray): A 2D array specifying the min and max for each input
                                 dimension. Shape (input_dim, 2), where ranges[i, 0] is
                                 the minimum and ranges[i, 1] is the maximum for the i-th dimension.

        Output:
            np.ndarray: The standardized input data. Shape (n_samples, input_dim).
        """
        X = np.asarray(X, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        mins = ranges[:, 0]
        maxs = ranges[:, 1]
        ranges_width = maxs - mins
        ranges_width[ranges_width == 0] = 1.0
        X_standardized = (X - mins) / ranges_width
        return np.clip(X_standardized, 0.0, 1.0)

    def _standardize_output(self, Y):
        """
        Standardizes the output data Y by centering and scaling.

        Arguments:
            Y (np.ndarray): The output data to be standardized. Shape (n_samples, output_dim).

        Output:
            np.ndarray: The standardized output data. Shape (n_samples, output_dim).
        """

        self.standardization_mean = np.mean(Y, axis=0)
        Y_centered = Y - self.standardization_mean
        self.standardization_scale = np.sqrt(np.mean(Y_centered ** 2, axis=0))
        return Y_centered / self.standardization_scale

    def _unstandardize_output(self, Y_standardized):
        """
        Unstandardizes the output data Y_standardized using the previously stored mean and scale.

        Arguments:
            Y_standardized (np.ndarray): The standardized output data to be unstandardized.
                                         Shape (n_samples, output_dim).

        Output:
            np.ndarray: The unstandardized output data. Shape (n_samples, output_dim).
        """
        return Y_standardized * self.standardization_scale + self.standardization_mean

    def compute_principal_components(self, Y_standardized):
        """
        Computes the principal component weights and basis vectors from standardized output data.

        Arguments:
            Y_standardized (tf.Tensor or np.ndarray): The standardized training output data.
                                                      Shape (n_samples, output_dim).

        Output:
            tuple: containing:
                - weights (np.ndarray): The principal component weights.
                                        Shape (n_samples, n_components).
                - phi_basis (np.ndarray): The principal component basis vectors.
                                          Shape (output_dim, n_components).
        """
        y_tensor = tf.convert_to_tensor(Y_standardized, dtype=tf.float64)
        
        n = tf.cast(tf.shape(y_tensor)[0], dtype=tf.float64) 
        m = tf.cast(tf.shape(y_tensor)[1], dtype=tf.float64) 
        s, u, v = tf.linalg.svd(y_tensor, full_matrices=False)
        q = self.n_components
        
        weights_raw = (1 / tf.sqrt(n)) * tf.matmul(u, tf.linalg.diag(s))[:, :q]
        
        std_devs = tf.math.reduce_std(weights_raw, axis=0)
        
        weights = weights_raw / std_devs  # (n x q)
        phi_basis = tf.sqrt(n) * tf.matmul(v[:, :q], tf.linalg.diag(std_devs))  # (m x q)
        
        return weights.numpy(), phi_basis.numpy()

    def _build_kernel_matrix(self, X1, X2=None, component_idx=None):
        """
        Computes a Gaussian Kernel covariance matrix for given input data.

        Arguments:
            X1 (tf.Tensor or np.ndarray): The first set of input points. Shape (n1, input_dim).
            X2 (tf.Tensor or np.ndarray, optional): The second set of input points. Shape (n2, input_dim).
            component_idx (int, optional): Builds the kernel matrix only for this
                                           particular principal component (0-indexed).
                                           If None, a block-diagonal matrix for all components is built.

        Output:
            tf.Tensor: The kernel (covariance) matrix.
                       - If `component_idx` is specified: Shape (n1, n2).
                       - If `component_idx` is None: Shape (n1 * n_components, n2 * n_components)
                                                     (block-diagonal).
        """
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

    def _negative_log_marginal_likelihood(self, rho_flattened, lambda_w, noise_var):
        """
        Calculate the negative log marginal likelihood for PCGP model using stored matrices.

        Args:
            rho_flattened (np.ndarray): Flattened array of length scales (n_components * input_dim)
            lambda_w (np.ndarray): Array of precision parameters (n_components)
            noise_var (float): Noise variance parameter

        Returns:
            float: Negative log marginal likelihood value
        """
        # Update hyperparameters and recompute matrices if they changed
        self.rho = np.reshape(rho_flattened, (self.n_components, self.input_dim))
        self.lambda_w = lambda_w
        self.noise_var = noise_var
        
        if self._hyperparams_changed(rho_flattened, lambda_w, noise_var):
            self._compute_and_store_matrices()

        n = self.X_train_std.shape[0]
        total_nll = 0

        for k in range(self.n_components):
            w_k = tf.constant(self.weights[:, k:k+1], dtype=tf.float64)
            
            # Use stored matrices
            L_k = self._stored_cholesky[k]

            # term 1
            alpha_k = tf.linalg.cholesky_solve(L_k, w_k)
            data_fit_k = tf.squeeze(tf.matmul(tf.transpose(w_k), alpha_k))

            # term 2
            log_det_k = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_k)))

            # term 3
            constant = n * np.log(2.0 * np.pi)

            nll_k = 0.5 * (data_fit_k + log_det_k + constant)
            total_nll += nll_k

        return tf.cast(total_nll, dtype=tf.float64).numpy()

    def fit(self, X_train, Y_train, ranges):
        """
        Fits the PCGP model to training data by optimizing hyperparameters.

        Arguments:
            X_train (np.ndarray): The training input data. Shape (n_train_samples, input_dim).
            Y_train (np.ndarray): The training output data. Shape (n_train_samples, output_dim).
            ranges (np.ndarray): The min/max ranges for standardizing input data.
                                 Shape (input_dim, 2).

        Output:
            self: The fitted PrincipalComponentGaussianProcessModel model.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_std = self.standardize_inputs(X_train, ranges)
        self.Y_train_std = self._standardize_output(Y_train)
        self.weights, self.phi_basis = self.compute_principal_components(self.Y_train_std)
        
        self._last_hyperparams = None
        
        iteration_count = [0]
        
        def objective(params):
            """
            Objective function for scipy.optimize.minimize.
            """
            iteration_count[0] += 1
            
            rho_flattened = np.exp(params[:self.n_components * self.input_dim])
            lambda_w = np.exp(params[self.n_components * self.input_dim : -1])
            noise_var = np.exp(params[-1])
            
            nll = self._negative_log_marginal_likelihood(rho_flattened, lambda_w, noise_var)
            
            # if iteration_count[0] % 10 == 0:
            #     print(f"Iteration {iteration_count[0]}: NLL = {nll:.6f}, noise_var = {noise_var:.6f}")
            #     print(f"  Sample rho: {rho_flattened[:3]}")
            #     print(f"  Sample lambda_w: {lambda_w[:3]}")
            
            return nll

        initial_rho_log = np.log(self.rho.flatten())
        initial_lambda_w_log = np.log(self.lambda_w)
        initial_noise_var_log = np.log(self.noise_var)

        initial_params_flat = np.concatenate([
            initial_rho_log,
            initial_lambda_w_log,
            np.array([initial_noise_var_log])
        ])

        bounds = []
        for _ in range(self.n_components * self.input_dim):
            bounds.append((-10, 5))  
        for _ in range(self.n_components):
            bounds.append((-10, 5))  
        bounds.append((-15, 0))  

        result = minimize(
            fun=objective,
            x0=initial_params_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        print(f"Optimization completed in {iteration_count[0]} iterations")
        print(f"Final NLL: {result.fun:.6f}")
        
        opt_params = result.x
        self.rho = np.exp(np.clip(opt_params[:self.n_components * self.input_dim], -10, 5))
        self.rho = np.reshape(self.rho, (self.n_components, self.input_dim))
        self.lambda_w = np.exp(np.clip(opt_params[self.n_components * self.input_dim:-1], -10, 5))
        self.noise_var = np.exp(np.clip(opt_params[-1], -15, 0))

        for i in range(self.n_components):
            print(f"Component {i+1}:")
            print(f"  Length scales (ρ): {self.rho[i]}")
            print(f"  Precision (λ): {self.lambda_w[i]:.4f}")
        print(f"Noise variance: {self.noise_var:.6f}")

        return self

    def predict(self, X_new, ranges, return_std=False, debug=True):
        """
        Makes predictions using the fitted PCGP model with stored matrices.

        Arguments:
            X_new (np.ndarray): The new input data for which to make predictions.
                                Shape (n_test_samples, input_dim).
            ranges (np.ndarray): The min/max ranges used for standardizing input data.
                                 Shape (input_dim, 2).
            return_std (bool, optional): If True, also returns the standard deviation of the
                                         predictions. Defaults to False.
            debug (bool, optional): A placeholder for potential debug outputs.

        Output:
            np.ndarray: The mean predictions for the new input data. Shape (n_test_samples, output_dim).
            (np.ndarray, optional): If `return_std` is True, also returns the standard deviation
                                    of the predictions. Shape (n_test_samples, output_dim).
        """
        X_new_std = self.standardize_inputs(X_new, ranges)
        X_new_tf = tf.convert_to_tensor(X_new_std, dtype=tf.float64)
        X_train_tf = tf.convert_to_tensor(self.X_train_std, dtype=tf.float64)
        
        n_train = self.X_train_std.shape[0]
        n_test = X_new_std.shape[0]
        
        mu_g = np.zeros((n_test, self.n_components))
        
        var_y_std = np.zeros((n_test, self.output_dim)) 

        for k in range(self.n_components):
            w_k = tf.constant(self.weights[:, k:k+1], dtype=tf.float64)
            phi_k = tf.constant(self.phi_basis[:, k:k+1], dtype=tf.float64)
            
            L_k = self._stored_cholesky[k]
            k_star = self._build_kernel_matrix(X_new_tf, X_train_tf, component_idx=k)
            
            alpha_k = tf.linalg.cholesky_solve(L_k, w_k)
            mu_k = tf.matmul(k_star, alpha_k)
            mu_g[:, k] = tf.squeeze(mu_k).numpy()
            
            if return_std:
                k_star_star = self._build_kernel_matrix(X_new_tf, component_idx=k)
                v_k = tf.linalg.cholesky_solve(L_k, tf.transpose(k_star))  

                Cov_k = k_star_star - tf.matmul(k_star, v_k)

                diag_Cov_k = tf.linalg.diag_part(Cov_k).numpy() 

                var_contrib_k = np.outer(diag_Cov_k, self.phi_basis[:, k]**2) 
                var_y_std += var_contrib_k

        phi_tf = tf.convert_to_tensor(self.phi_basis, dtype=tf.float64)
        mu_y_std = tf.matmul(mu_g, phi_tf, transpose_b=True)
        mean_y = self._unstandardize_output(mu_y_std.numpy())
        
        if not return_std:
            return mean_y
        
        var_y_std += self.noise_var 

        var_y = var_y_std * (self.standardization_scale ** 2)
        std_y = np.sqrt(var_y)
        
        return mean_y, std_y
    
