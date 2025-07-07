
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
        Phi_basis = tf.transpose(v[:actual_components, :]).numpy()  # Transpose to get (output_dim, n_components)
        
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

    def _tf_kron(self, A, B):
        """
        Computes the Kronecker product of two matrices A and B using TensorFlow.
        """
        s1, s2 = A.shape
        s3, s4 = B.shape
        return tf.reshape(tf.einsum('ij,kl->ikjl', A, B), [s1 * s3, s2 * s4])


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
        self.rho = np.reshape(rho_flattened, (self.n_components, self.input_dim))
        self.lambda_w = lambda_w
        self.noise_var = noise_var

        m = self.X_train_std.shape[0] 
        q = self.n_components
        n = self.output_dim 

        Phi_tf = tf.constant(self.Phi_basis, dtype=tf.float64) # (n x q)

        Sigma_YY = tf.zeros((m * n, m * n), dtype=tf.float64)
        for j in range(q):
            K_j_XX = self._build_kernel_matrix(self.X_train_std, component_idx=j) # (m x m)

            Phi_j = Phi_tf[:, j][:, None] # (n x 1)

            term_j = self._tf_kron(tf.matmul(Phi_j, tf.transpose(Phi_j)), K_j_XX) # (mn x mn)
            Sigma_YY += term_j

        noise_term = tf.eye(m * n, dtype=tf.float64) * self.noise_var
        Sigma_YY += noise_term 

        L_Sigma_YY = tf.linalg.cholesky(Sigma_YY)

        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Sigma_YY)))
        Y_flat = tf.constant(self.Y_train_std.flatten(), dtype=tf.float64)[:, None]
        alpha = tf.linalg.cholesky_solve(L_Sigma_YY, Y_flat)
        data_fit = tf.squeeze(tf.matmul(tf.transpose(Y_flat), alpha))

        constant_val = 0.5 * m * n * np.log(2.0 * np.pi)

        return (tf.cast(0.5, dtype=tf.float64) * data_fit +
                tf.cast(0.5, dtype=tf.float64) * log_det +
                tf.cast(constant_val, dtype=tf.float64)).numpy()


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
        self.K_eta_scores, self.Phi_basis = self.compute_principal_components(self.Y_train_std)

        def objective(params):
            rho_flattened = np.exp(params[:self.n_components * self.input_dim])
            lambda_w = np.exp(params[self.n_components * self.input_dim : self.n_components * self.input_dim + self.n_components])
            noise_var = np.exp(params[-1])
            return self._negative_log_marginal_likelihood(rho_flattened, lambda_w, noise_var)

        initial_rho_log = np.log(self.rho.flatten())
        initial_lambda_w_log = np.log(self.lambda_w)
        initial_noise_var_log = np.log(self.noise_var)

        initial_params_flat = np.concatenate([
            initial_rho_log,
            initial_lambda_w_log,
            np.array([initial_noise_var_log])
        ])

        bounds = [(None, None)] * len(initial_params_flat)

        result = minimize(
                fun=objective,
                x0=initial_params_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000}
        )

        opt_params = result.x
        self.rho = np.exp(np.reshape(opt_params[:self.n_components * self.input_dim],
                            (self.n_components, self.input_dim)))
        self.lambda_w = np.exp(opt_params[self.n_components * self.input_dim:-1])
        self.noise_var = np.exp(opt_params[-1])

        for i in range(self.n_components):
            print(f"Component {i+1}:")
            print(f"  Length scales (ρ): {self.rho[i]}")
            print(f"  Precision (λ): {self.lambda_w[i]:.4f}")
        print(f"Noise variance: {self.noise_var:.6f}")

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

        Sigma_YY = tf.zeros((N * n, N * n), dtype=tf.float64)
        Phi_tf = tf.constant(self.Phi_basis, dtype=tf.float64)

        for j in range(q):
            K_j_XX = self._build_kernel_matrix(self.X_train_std, component_idx=j)
            Phi_j = Phi_tf[:, j][:, None]
            term_j = self._tf_kron(tf.matmul(Phi_j, tf.transpose(Phi_j)), K_j_XX)
            Sigma_YY += term_j
        Sigma_YY += tf.eye(N * n, dtype=tf.float64) * self.noise_var
        L_Sigma_YY = tf.linalg.cholesky(Sigma_YY)

        Y_cent_flat = tf.constant(self.Y_train_std.flatten(), dtype=tf.float64)[:, None]

        alpha = tf.linalg.cholesky_solve(L_Sigma_YY, Y_cent_flat)

        K_XnewX_matrix = tf.zeros((N_test * n, N * n), dtype=tf.float64)

        for j in range(q):
            K_j_XnewX = self._build_kernel_matrix(X_new_std, self.X_train_std, component_idx=j) # (N_test x N)
            Phi_j = Phi_tf[:, j][:, None]
            term_j = self._tf_kron(tf.matmul(Phi_j, tf.transpose(Phi_j)), K_j_XnewX)
            K_XnewX_matrix += term_j

        mu_f_flat_std = tf.matmul(K_XnewX_matrix, alpha)
        mu_f_std = tf.reshape(mu_f_flat_std, (N_test, n)).numpy()
        pred_mean = self._unstandardize_output(mu_f_std)

        if return_std:
            # Calculate K(X*,X*)
            K_XnewXnew_matrix = tf.zeros((N_test * n, N_test * n), dtype=tf.float64)
            for j in range(q):
                K_j_XnewXnew = self._build_kernel_matrix(X_new_std, component_idx=j) # (N_test x N_test)
                Phi_j = Phi_tf[:, j][:, None]
                term_j = self._tf_kron(tf.matmul(Phi_j, tf.transpose(Phi_j)), K_j_XnewXnew)
                K_XnewXnew_matrix += term_j

            K_fF = K_XnewX_matrix
            K_ff = K_XnewXnew_matrix

            v = tf.linalg.cholesky_solve(L_Sigma_YY, tf.transpose(K_fF))

            cov_f = K_ff - tf.matmul(K_fF, v)

            cov_y = cov_f + tf.eye(N_test * n, dtype=tf.float64) * self.noise_var

            pred_var = np.diag(cov_y.numpy()).reshape(N_test, n)
            pred_var = np.maximum(pred_var, 0) 
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

        scaled_sq_diff = tf.square(X1[:, None, :] - X2[None, :, :]) / tf.square(rho_safe[None, None, :])
        exponent = -0.5 * tf.reduce_sum(scaled_sq_diff, axis=-1)
        return self.variance * tf.exp(exponent)

    def set_hyperparameters(self, variance, rho=None):
        self.variance.assign(variance)
        if rho is not None:
            self.rho.assign(rho)


# AI GENERATED TEST (re-run with fixed class)
def generate_test_data(n_train=50, n_test=20, input_dim=3, output_dim=5):
    np.random.seed(42)

    X_train = np.random.uniform(0, 1, (n_train, input_dim))
    ranges = [(0, 1)] * input_dim

    def true_func(x):
        y1 = (x**2).sum(axis=1)
        y2 = np.sin(x[:, 0]*10) + np.cos(x[:, 1]*5)
        y3 = np.exp(x[:, 2])
        y_outputs = [y1, y2, y3]

        if output_dim > len(y_outputs):
            for i in range(len(y_outputs), output_dim):
                y_outputs.append(np.sin(x[:, (i % input_dim)] * (i + 1) * 5))
        elif output_dim < len(y_outputs):
            y_outputs = y_outputs[:output_dim]

        return np.stack(y_outputs, axis=1)

    Y_train = true_func(X_train)

    Y_train += np.random.normal(0, 0.05, Y_train.shape) # Reduced noise for clearer results

    X_test = np.random.uniform(0, 1, (n_test, input_dim))
    Y_test = true_func(X_test)

    return X_train, Y_train, X_test, Y_test, ranges, true_func

n_train_val = 50
input_dim_val = 3
output_dim_val = 5
X_train, Y_train, X_test_dummy, Y_test_dummy, ranges, true_func_global = generate_test_data(n_train=n_train_val, input_dim=input_dim_val, output_dim=output_dim_val)

print("Fitting your PCGP model...")
your_pcgp = PrincipalComponentGaussianProcessModel(n_components=min(output_dim_val, X_train.shape[0]-1), input_dim=X_train.shape[1], output_dim=Y_train.shape[1])
your_pcgp.fit(X_train, Y_train, ranges)


# input dimension
input_to_vary_idx = 1 # Let's vary the second input dimension

for output_idx in range(output_dim_val):
    num_plot_points = 200
    x_min, x_max = ranges[input_to_vary_idx]
    X_plot = np.zeros((num_plot_points, input_dim_val))

    # Set the input dimension to vary
    X_plot[:, input_to_vary_idx] = np.linspace(x_min, x_max, num_plot_points)

    # Set other input dimensions to their training data mean
    for dim_idx in range(input_dim_val):
        if dim_idx != input_to_vary_idx:
            X_plot[:, dim_idx] = np.mean(X_train[:, dim_idx])

    your_pred_mean, your_pred_std = your_pcgp.predict(X_plot, ranges, return_std=True)

    Y_true_plot = true_func_global(X_plot)

    plt.figure(figsize=(10, 6))

    # Plot training observations for the specific output dimension
    plt.plot(X_train[:, input_to_vary_idx], Y_train[:, output_idx], 'kx', alpha=0.6, label='Training Observations')

    # Plot predicted mean
    plt.plot(X_plot[:, input_to_vary_idx], your_pred_mean[:, output_idx], 'b-', label='Predicted mean')

    # Plot true function
    plt.plot(X_plot[:, input_to_vary_idx], Y_true_plot[:, output_idx], 'r-', alpha=0.6, label='True function')

    # Plot 95% confidence interval
    mean_np = your_pred_mean[:, output_idx]
    std_np = your_pred_std[:, output_idx]

    plt.fill_between(X_plot[:, input_to_vary_idx].flatten(),
                    (mean_np - 2 * std_np),
                    (mean_np + 2 * std_np),
                    alpha=0.2, color='blue', label='95% Confidence Interval')

    plt.xlabel(f'Input X (Dimension {input_to_vary_idx})')
    plt.ylabel(f'Output Y (Dimension {output_idx})')
    plt.title(f'Your PCGP Regression (Output {output_idx})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()