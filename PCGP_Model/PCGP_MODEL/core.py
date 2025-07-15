import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PCGP_MODEL.kernels import GaussianKernel

class PrincipalComponentGaussianProcessModel:
    def __init__(self, n_components=9, input_dim=12, output_dim=28):
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
        # maybe store alpha

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
        self.standardization_scale = np.sqrt(np.mean(Y_centered ** 2, axis=0))
        # if self.standardization_scale == 0:
        #     self.standardization_scale = 1.0
        return Y_centered / self.standardization_scale

    def _unstandardize_output(self, Y_standardized):
        return Y_standardized * self.standardization_scale + self.standardization_mean

    # weights variance not 1
    # def compute_principal_components(self, Y_standardized):
    #     y_tensor = tf.convert_to_tensor(Y_standardized, dtype=tf.float64)
        
    #     n = tf.cast(tf.shape(y_tensor)[0], dtype=tf.float64)
        
    #     s, u, v = tf.linalg.svd(y_tensor, full_matrices=False)
        
    #     q = self.n_components
        
    #     weights = 1 / tf.sqrt(n) * tf.matmul(u, tf.linalg.diag(s))[:, :q]  # (n x q)
    #     phi_basis = tf.sqrt(n) * tf.transpose(v)[:, :q]  # (m x q)
        
    #     return weights.numpy(), phi_basis.numpy()

    def compute_principal_components(self, Y_standardized):
        y_tensor = tf.convert_to_tensor(Y_standardized, dtype=tf.float64)
        
        n = tf.cast(tf.shape(y_tensor)[0], dtype=tf.float64) 
        m = tf.cast(tf.shape(y_tensor)[1], dtype=tf.float64) 

        s, u, v = tf.linalg.svd(y_tensor, full_matrices=False)
        
        q = self.n_components
        
        u_q = u[:, :q]
        s_q = tf.linalg.diag(s[:q])
        v_q = v[:, :q] 

        w_raw = tf.matmul(u_q, s_q) # (n x q)
        
        std_devs = tf.math.reduce_std(w_raw, axis=0)

        weights = w_raw / std_devs # (n x q)

        phi_basis = tf.matmul(v_q, tf.linalg.diag(std_devs)) # (m x q)

        return weights.numpy(), phi_basis.numpy()

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

    def _negative_log_marginal_likelihood(self, rho_flattened, lambda_w, noise_var):
        """
        Calculate the negative log marginal likelihood for PCGP model using _build_kernel_matrix.

        Args:
            rho_flattened (np.ndarray): Flattened array of length scales (n_components * input_dim)
            lambda_w (np.ndarray): Array of precision parameters (n_components)
            noise_var (float): Noise variance parameter

        Returns:
            float: Negative log marginal likelihood value
        """
        self.rho = np.reshape(rho_flattened, (self.n_components, self.input_dim))
        self.lambda_w = lambda_w
        self.noise_var = noise_var

        m = self.output_dim 
        n = self.X_train_std.shape[0] 
        q = self.n_components

        total_nll = 0

        for k in range(self.n_components):
            w_k = tf.constant(self.weights[:, k:k+1], dtype=tf.float64)
            K_k = self._build_kernel_matrix(self.X_train_std, component_idx=k)

            Sigma_k = K_k + self.noise_var * tf.eye(n, dtype=tf.float64)
            L_k = tf.linalg.cholesky(Sigma_k)

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
        Fits the PCGP model to training data
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_std = self.standardize_inputs(X_train, ranges)
        self.Y_train_std = self._standardize_output(Y_train)
        self.weights, self.phi_basis = self.compute_principal_components(self.Y_train_std)
        
        iteration_count = [0]
        
        def objective(params):
            iteration_count[0] += 1
            
            rho_flattened = np.exp(params[:self.n_components * self.input_dim])
            lambda_w = np.exp(params[self.n_components * self.input_dim : -1])
            noise_var = np.exp(params[-1])
            
            nll = self._negative_log_marginal_likelihood(rho_flattened, lambda_w, noise_var)
            
            # debug
            if iteration_count[0] % 10 == 0:
                print(f"Iteration {iteration_count[0]}: NLL = {nll:.6f}, noise_var = {noise_var:.6f}")
                print(f"  Sample rho: {rho_flattened[:3]}")
                print(f"  Sample lambda_w: {lambda_w[:3]}")
            
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

    def predict(self, X_new, ranges, return_std=False):
        """
        Makes predictions using the fitted PCGP model according to Theorems 3.1, 3.3 and 3.4.
        
        Args:
            X_new (np.ndarray): New input locations to predict at (n_test x input_dim).
            ranges (list): List of (min, max) tuples for each parameter for standardizing inputs.
            return_std (bool): If True, returns both mean and standard deviation of predictions.
        
        Returns:
            np.ndarray: Predicted mean values at X_new (n_test x output_dim).
            (np.ndarray, np.ndarray): If return_std=True, returns tuple of (mean, std) 
                                    where std has shape (n_test x output_dim).
        """
        X_new_std = self.standardize_inputs(X_new, ranges)
        X_new_tf = tf.convert_to_tensor(X_new_std, dtype=tf.float64)
        X_train_tf = tf.convert_to_tensor(self.X_train_std, dtype=tf.float64)
        
        n_train = self.X_train_std.shape[0]
        n_test = X_new_std.shape[0]
        
        mu_g = np.zeros((n_test, self.n_components))  
        var_g = np.zeros((n_test, self.n_components)) 
        
        for k in range(self.n_components):
            # from therom 3.1
            w_k = self.weights[:, k:k+1]  # mk
            
            C_k = self._build_kernel_matrix(X_train_tf, component_idx=k)
            
            # sk
            d_k = 1.0 / self.lambda_w[k]
            C_k_inv = tf.linalg.inv(C_k)
            S_k = tf.linalg.inv(d_k * tf.eye(n_train, dtype=tf.float64) + C_k_inv)
            
            # from theorem 3.3
            c_k_x = self._build_kernel_matrix(X_new_tf, X_train_tf, component_idx=k)
            
            c_k_xx_full = self._build_kernel_matrix(X_new_tf, component_idx=k)
            c_k_xx_diag = tf.linalg.diag_part(c_k_xx_full)
            
            # tk
            C_k_inv_Sk = tf.matmul(C_k_inv, S_k)
            T_k = C_k_inv - tf.matmul(C_k_inv_Sk, C_k_inv)

            # u_k(x)
            alpha = tf.linalg.cholesky_solve(tf.linalg.cholesky(C_k), w_k)
            mu_k = tf.matmul(c_k_x, alpha)
            mu_g[:, k] = tf.squeeze(mu_k).numpy()
            
            # var_k(x)
            v = tf.matmul(T_k, tf.transpose(c_k_x))
            var_k = c_k_xx_diag - tf.reduce_sum(c_k_x * tf.transpose(v), axis=1)
            var_g[:, k] = tf.maximum(var_k, 1e-12).numpy()
        
        # from theorem 3.4
        phi_tf = tf.convert_to_tensor(self.phi_basis, dtype=tf.float64)
        mu_y_std = tf.matmul(mu_g, phi_tf, transpose_b=True)
        
        # u_g(x)
        mean_y = self._unstandardize_output(mu_y_std.numpy())
        
        if not return_std:
            return mean_y
        
        # cov_y(x)
        phi_sq = tf.square(phi_tf)  
        var_y_std = tf.matmul(var_g, phi_sq, transpose_b=True)
        var_y_std += self.noise_var  
        
        std_y = np.sqrt(var_y_std) * self.standardization_scale
        
        return mean_y, std_y
