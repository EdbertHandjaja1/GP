import numpy as np
import tensorflow as tf
import scipy as scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import gpflow

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
        
        phi_basis = tf.transpose(v[:actual_components, :]).numpy()
        
        weights = (u[:, :actual_components] @ tf.linalg.diag(s[:actual_components])).numpy()

        # phi_basis = v[:actual_components, :].numpy().T  # (output_dim, n_components)
        
        # weights = (Y_standardized @ phi_basis).astype(np.float64)
        
        return weights, phi_basis
    
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

    def _build_kernel_matrix_reorganized(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        n1 = X1.shape[0]
        n2 = X2.shape[0]
        q = self.n_components
        
        K_full = np.zeros((n1 * q, n2 * q))
        
        for i in range(n1):
            for j in range(n2):
                block = np.zeros((q, q))
                for k in range(q):
                    variance = 1.0 / self.lambda_w[k]
                    rho = self.rho[k, :]
                    kernel = GaussianKernel(variance=variance, rho=rho, input_dim=self.input_dim)
                    
                    xi = tf.expand_dims(X1[i], 0)
                    xj = tf.expand_dims(X2[j], 0)
                    k_val = kernel(xi, xj)[0, 0]
                    
                    block[k, k] = k_val
                
                K_full[i*q:(i+1)*q, j*q:(j+1)*q] = block
        
        return tf.convert_to_tensor(K_full, dtype=tf.float64)

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
            
        K_full = self._build_kernel_matrix_reorganized(self.X_train_std)
        
        phi_tf = tf.constant(self.phi_basis, dtype=tf.float64) 

        I_n = tf.eye(n, dtype=tf.float64)
        kron_I_phi_operator = tf.linalg.LinearOperatorKronecker([
            tf.linalg.LinearOperatorFullMatrix(I_n), 
            tf.linalg.LinearOperatorFullMatrix(phi_tf)
        ])
        
        kron_I_phi_dense = kron_I_phi_operator.to_dense()

        Sigma_YY = tf.matmul(kron_I_phi_dense, K_full)
        Sigma_YY = tf.matmul(Sigma_YY, tf.transpose(kron_I_phi_dense))
            
        noise_term = tf.eye(m * n, dtype=tf.float64) * self.noise_var
        Sigma_YY += noise_term
            
        L_Sigma_YY = tf.linalg.cholesky(Sigma_YY)
            
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Sigma_YY)))
            
        Y_flat = tf.constant(self.Y_train_std.flatten('F'), dtype=tf.float64)[:, None]
            
        alpha = tf.linalg.cholesky_solve(L_Sigma_YY, Y_flat)
        data_fit = tf.squeeze(tf.matmul(tf.transpose(Y_flat), alpha))
        constant_val = 0.5 * m * n * np.log(2.0 * np.pi)
            
        nll = (0.5 * data_fit + 0.5 * log_det + constant_val)

        return tf.cast(nll, dtype=tf.float64).numpy()
        ### for nll test ###
        # return tf.cast(nll, dtype=tf.float64).numpy(), K_full
    
    def fit(self, X_train, Y_train, ranges):
        """
        Fits the PCGP model to training data
        """
        self.X_train = X_train
        self.X_train_std = self.standardize_inputs(X_train, ranges)
        self.Y_train_std = self._standardize_output(Y_train)
        self.weights, self.phi_basis = self.compute_principal_components(self.Y_train_std)
        
        # iteration_count = [0]
        
        # def objective(params):
        #     iteration_count[0] += 1
            
        #     rho_flattened = np.exp(np.clip(params[:self.n_components * self.input_dim], -10, 5))
        #     lambda_w = np.exp(np.clip(params[self.n_components * self.input_dim : self.n_components * self.input_dim + self.n_components], -10, 5))
        #     noise_var = np.exp(np.clip(params[-1], -15, 0))
            
        #     nll = self._negative_log_marginal_likelihood(rho_flattened, lambda_w, noise_var)
            
        #     # debug
        #     if iteration_count[0] % 10 == 0:
        #         print(f"Iteration {iteration_count[0]}: NLL = {nll:.6f}, noise_var = {noise_var:.6f}")
        #         print(f"  Sample rho: {rho_flattened[:3]}")
        #         print(f"  Sample lambda_w: {lambda_w[:3]}")
            
        #     return nll
        
        # initial_rho_log = np.log(np.clip(self.rho.flatten(), 1e-6, 10))
        # initial_lambda_w_log = np.log(np.clip(self.lambda_w, 1e-6, 10))
        # initial_noise_var_log = np.log(np.clip(self.noise_var, 1e-15, 1))

        # initial_params_flat = np.concatenate([
        #     initial_rho_log,
        #     initial_lambda_w_log,
        #     np.array([initial_noise_var_log])
        # ])
        
        # bounds = []
        # for _ in range(self.n_components * self.input_dim):
        #     bounds.append((-10, 5))  
        # for _ in range(self.n_components):
        #     bounds.append((-10, 5))  
        # bounds.append((-15, 0))  

        # result = minimize(
        #     fun=objective,
        #     x0=initial_params_flat,
        #     method='L-BFGS-B',
        #     bounds=bounds,
        #     options={'disp': True}
        # )
        
        # print(f"Optimization completed in {iteration_count[0]} iterations")
        # print(f"Final NLL: {result.fun:.6f}")
        
        # opt_params = result.x
        # self.rho = np.exp(np.clip(opt_params[:self.n_components * self.input_dim], -10, 5))
        # self.rho = np.reshape(self.rho, (self.n_components, self.input_dim))
        # self.lambda_w = np.exp(np.clip(opt_params[self.n_components * self.input_dim:-1], -10, 5))
        # self.noise_var = np.exp(np.clip(opt_params[-1], -15, 0))

        # for i in range(self.n_components):
        #     print(f"Component {i+1}:")
        #     print(f"  Length scales (ρ): {self.rho[i]}")
        #     print(f"  Precision (λ): {self.lambda_w[i]:.4f}")
        # print(f"Noise variance: {self.noise_var:.6f}")

        # return self

        self.noise_var = 0.808792

        self.rho[0, :3] = [0.10165914, 0.03661507, 0.00734365]
        self.lambda_w[0] = 148.4132
        
        self.rho[1, :3] = [4.53999298e-05, 4.53999298e-05, 4.53999298e-05]
        self.lambda_w[1] = 1.5218
        
        self.rho[2, :3] = [3.01743863e-03, 8.97942316e-03, 4.53999298e-05]
        self.lambda_w[2] = 3.3930

        return self

    # def predict(self, X_new, ranges, return_std=False):
    #     """
    #     Makes predictions using the fitted PCGP model.

    #     Args:
    #         X_new (np.ndarray): New input locations to predict at (m_test x p).
    #         ranges (list): List of (min,max) tuples for each parameter for standardizing inputs.
    #         return_std (bool): If True, returns both mean and standard deviation of predictions.

    #     Returns:
    #         np.ndarray: Predicted mean values at X_new (m_test x n).
    #         (np.ndarray, np.ndarray): If return_std=True, returns tuple of (mean, std) where std has shape (m_test x n).
    #     """
    #     X_new_std = self.standardize_inputs(X_new, ranges)
    #     X_new_tf = tf.convert_to_tensor(X_new_std, dtype=tf.float64)
    #     X_train_tf = tf.convert_to_tensor(self.X_train_std, dtype=tf.float64)

    #     n_train = self.X_train_std.shape[0]
    #     n_test = X_new_std.shape[0]
    #     m = self.output_dim
    #     q = self.n_components
        
    #     K_train = self._build_kernel_matrix_reorganized(X_train_tf)  
    #     K_test_train = self._build_kernel_matrix_reorganized(X_new_tf, X_train_tf)  
    #     K_test = self._build_kernel_matrix_reorganized(X_new_tf)  
        
    #     phi_tf = tf.constant(self.phi_basis, dtype=tf.float64)
        
    #     I_n_train = tf.eye(n_train, dtype=tf.float64)
    #     kron_I_phi_train = tf.linalg.LinearOperatorKronecker([
    #         tf.linalg.LinearOperatorFullMatrix(I_n_train), 
    #         tf.linalg.LinearOperatorFullMatrix(phi_tf)
    #     ]).to_dense()
        
    #     I_n_test = tf.eye(n_test, dtype=tf.float64)
    #     kron_I_phi_test = tf.linalg.LinearOperatorKronecker([
    #         tf.linalg.LinearOperatorFullMatrix(I_n_test), 
    #         tf.linalg.LinearOperatorFullMatrix(phi_tf)
    #     ]).to_dense()
        
    #     Sigma_YY_train = tf.matmul(kron_I_phi_train, K_train)
    #     Sigma_YY_train = tf.matmul(Sigma_YY_train, tf.transpose(kron_I_phi_train))
    #     Sigma_YY_train += tf.eye(m * n_train, dtype=tf.float64) * self.noise_var
        
    #     Sigma_YY_test = tf.matmul(kron_I_phi_test, K_test)
    #     Sigma_YY_test = tf.matmul(Sigma_YY_test, tf.transpose(kron_I_phi_test))
        
    #     Sigma_YY_cross = tf.matmul(kron_I_phi_test, K_test_train)
    #     Sigma_YY_cross = tf.matmul(Sigma_YY_cross, tf.transpose(kron_I_phi_train))
        
    #     L_train = tf.linalg.cholesky(Sigma_YY_train)
    #     Y_train_flat = tf.constant(self.Y_train_std.flatten('F'), dtype=tf.float64)[:, None]
        
    #     alpha = tf.linalg.cholesky_solve(L_train, Y_train_flat)
    #     mean_flat = tf.matmul(Sigma_YY_cross, alpha)
        
    #     mean_std = tf.reshape(mean_flat, [n_test, m], name='mean_reshape')
    #     mean = self._unstandardize_output(mean_std.numpy())
        
    #     if not return_std:
    #         return mean
        
    #     v = tf.linalg.cholesky_solve(L_train, tf.transpose(Sigma_YY_cross))
    #     var_flat = tf.linalg.diag_part(Sigma_YY_test - tf.matmul(Sigma_YY_cross, v))
    #     var = tf.reshape(var_flat, [n_test, m])
        
    #     var = var + self.noise_var
    #     std = tf.sqrt(tf.maximum(var, 1e-12)) * self.standardization_scale
        
    #     return mean, std.numpy()

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
