import numpy as np

def generate_test_data(n_train=50, n_test=200, input_dim=3, output_dim=5, 
                     function_type='default', noise_level=0.05, random_seed=42):
    """
    Generate test data with different function types.
    """
    np.random.seed(random_seed)
    X_train = np.random.uniform(0, 1, (n_train, input_dim))
    ranges = [(0, 1)] * input_dim
    
    def default_func(x):
        return np.column_stack((
            np.sin(x[:, 0] * 2),
            x[:, 1] ** 2,
            x[:, 0] * x[:, 2],
            np.cos(x[:, 1] + x[:, 2]),
            np.exp(x[:, 0])
        ))
    
    def polynomial_func(x):
        return np.column_stack((
            x[:, 0] ** 3 - 2 * x[:, 1] ** 2 + x[:, 2],
            0.5 * x[:, 0] * x[:, 1] + x[:, 1] * x[:, 2],
            x[:, 0] + x[:, 1] + x[:, 2],
            x[:, 0] ** 2 - x[:, 1] ** 2,
            3 * x[:, 0] - 2 * x[:, 1] + x[:, 2] ** 2
        ))
    
    def trigonometric_func(x):
        return np.column_stack((
            np.sin(2 * np.pi * x[:, 0]),
            np.cos(3 * np.pi * x[:, 1]),
            np.sin(4 * np.pi * x[:, 2]),
            np.cos(2 * np.pi * (x[:, 0] + x[:, 1])),
            np.sin(3 * np.pi * (x[:, 1] + x[:, 2]))
        ))
    
    def exponential_func(x):
        return np.column_stack((
            np.exp(2 * x[:, 0]),
            np.exp(-x[:, 1]),
            np.exp(x[:, 0] + x[:, 2]),
            np.exp(-(x[:, 1] ** 2)),
            np.exp(0.5 * (x[:, 0] + x[:, 1] + x[:, 2]))
        ))
    
    def multiplicative_func(x):
        return np.column_stack((
            x[:, 0] * x[:, 1] * x[:, 2],
            x[:, 0] * x[:, 1] + x[:, 1] * x[:, 2],
            (x[:, 0] + 1) * (x[:, 1] - 0.5) * x[:, 2],
            x[:, 0] * np.sqrt(x[:, 1] * x[:, 2]),
            np.log(1 + x[:, 0] * x[:, 1] * x[:, 2])
        ))
    
    def complex_func(x):
        return np.column_stack((
            np.sin(x[:, 0] * 2) * np.exp(-x[:, 1]),
            x[:, 0] ** 2 + np.cos(x[:, 1] * 3) * x[:, 2],
            np.log(1 + x[:, 0]) * np.sqrt(x[:, 1] + x[:, 2]),
            np.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2)) * np.sin(x[:, 2] * 4),
            (x[:, 0] + x[:, 1]) / (1 + x[:, 2] ** 2)
        ))
    
    func_map = {
        'default': default_func,
        'polynomial': polynomial_func,
        'trigonometric': trigonometric_func,
        'exponential': exponential_func,
        'multiplicative': multiplicative_func,
        'complex': complex_func
    }
    
    true_func = func_map[function_type]
    
    Y_train = true_func(X_train) + np.random.normal(0, noise_level, (n_train, output_dim))
    X_test = np.random.uniform(0, 1, (n_test, input_dim))
    Y_test = true_func(X_test) 
    
    return X_train, Y_train, X_test, Y_test, ranges, true_func 