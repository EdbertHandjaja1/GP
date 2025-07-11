import numpy as np

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