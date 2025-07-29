# import numpy as np
# import sys
# import os
# import time
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from PCGP_MODEL import PrincipalComponentGaussianProcessModel
# from surmise.emulation import emulator

# def run_pcgp(n_components, input_dim, output_dim_idx, X_train, Y_train, X_test):
#     """
#     Run PCGP model and return predictions.
    
#     Args:
#         n_components (int): Number of principal components.
#         input_dim (int): Dimensionality of the input space.
#         output_dim_idx (int): The index of the output dimension to model.
#         X_train (np.ndarray): Training input data.
#         Y_train (np.ndarray): Training output data.
#         X_test (np.ndarray): Test input data.

#     Returns:
#         Y_pred_mean: Predictive mean of shape (n_test, 1)
#         Y_pred_std: Predictive variance of shape (n_test, 1)
#     """
#     pcgp = PrincipalComponentGaussianProcessModel(
#         n_components=n_components,
#         input_dim=input_dim,
#         output_dim=1  
#     )

#     ranges = np.column_stack([np.zeros(input_dim), np.ones(input_dim)])

#     Y_train_copy = Y_train[:, output_dim_idx].copy().reshape(-1, 1)

#     start_train = time.time()
#     fitted_model = pcgp.fit(
#         X_train.copy(),
#         Y_train_copy,
#         ranges
#     )
#     end_train = time.time()
#     train_time = end_train - start_train

#     start_pred = time.time()
#     Y_pred_mean, Y_pred_std = fitted_model.predict(X_test, ranges, return_std=True)
#     end_pred = time.time()
#     pred_time = end_pred - start_pred

#     return Y_pred_mean, Y_pred_std, train_time, pred_time


# def run_surmise(n_components, input_dim, output_dim_idx, X_train, Y_train, X_test):
#     """
#     Run Surmise PCGP model and return predictions.
    
#     Args:
#         n_components (int): Number of principal components.
#         input_dim (int): Dimensionality of the input space.
#         output_dim_idx (int): The index of the output dimension to model.
#         X_train (np.ndarray): Training input data.
#         Y_train (np.ndarray): Training output data.
#         X_test (np.ndarray): Test input data.

#     Returns:
#         Y_pred_mean: Predictive mean of shape (n_test, 1)
#         Y_pred_std: Predictive variance of shape (n_test, 1)
#         emu: fitted emulator object
#     """
#     theta_emu_train = X_train.copy()
#     x_emu_train = np.array([[0]]) 
    
#     f_emu_train = Y_train[:, output_dim_idx].copy().reshape(1, -1)
    
#     emu = emulator(
#         x=x_emu_train, 
#         theta=theta_emu_train, 
#         f=f_emu_train, 
#         method='PCGP', 
#         options={'epsilon': 0}
#     )
#     start_train = time.time()
#     emu.fit()
#     end_train = time.time()
#     train_time = end_train - start_train
    
#     start_pred = time.time()
#     pred = emu.predict(x=x_emu_train, theta=X_test) 

#     Y_pred_mean = pred.mean().flatten()
#     Y_pred_std = np.sqrt(pred.var()).flatten()

#     end_pred = time.time()
#     pred_time = end_pred - start_pred
    
#     Y_pred_mean = Y_pred_mean.reshape(-1, 1)
#     Y_pred_std = Y_pred_std.reshape(-1, 1)
    
#     return Y_pred_mean, Y_pred_std, emu, train_time, pred_time

import numpy as np
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from surmise.emulation import emulator

def run_pcgp(n_components, input_dim, output_dim_idx, X_train, Y_train, X_test):
    """
    Run PCGP model and return predictions.
    
    Args:
        n_components (int): Number of principal components.
        input_dim (int): Dimensionality of the input space.
        output_dim_idx (int): The index of the output dimension to model.
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Test input data.

    Returns:
        Y_pred_mean: Predictive mean of shape (n_test, 1)
        Y_pred_std: Predictive variance of shape (n_test, 1)
    """
    # Set random seed for reproducible model initialization
    np.random.seed(0)  # Fixed seed for model consistency
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=1  
    )

    ranges = np.column_stack([np.zeros(input_dim), np.ones(input_dim)])

    Y_train_copy = Y_train[:, output_dim_idx].copy().reshape(-1, 1)
    
    # Normalize training data for better model performance
    Y_mean = np.mean(Y_train_copy)
    Y_std = np.std(Y_train_copy)
    if Y_std > 0:
        Y_train_normalized = (Y_train_copy - Y_mean) / Y_std
    else:
        Y_train_normalized = Y_train_copy
        Y_std = 1.0

    start_train = time.time()
    fitted_model = pcgp.fit(
        X_train.copy(),
        Y_train_normalized,
        ranges
    )
    end_train = time.time()
    train_time = end_train - start_train

    start_pred = time.time()
    Y_pred_mean_norm, Y_pred_std_norm = fitted_model.predict(X_test, ranges, return_std=True)
    end_pred = time.time()
    pred_time = end_pred - start_pred
    
    # Denormalize predictions
    Y_pred_mean = Y_pred_mean_norm * Y_std + Y_mean
    Y_pred_std = Y_pred_std_norm * Y_std

    return Y_pred_mean, Y_pred_std, train_time, pred_time


def run_surmise(n_components, input_dim, output_dim_idx, X_train, Y_train, X_test):
    """
    Run Surmise PCGP model and return predictions.
    
    Args:
        n_components (int): Number of principal components.
        input_dim (int): Dimensionality of the input space.
        output_dim_idx (int): The index of the output dimension to model.
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Test input data.

    Returns:
        Y_pred_mean: Predictive mean of shape (n_test, 1)
        Y_pred_std: Predictive variance of shape (n_test, 1)
        emu: fitted emulator object
    """
    # Set random seed for reproducible model initialization
    np.random.seed(0)  # Fixed seed for model consistency
    
    theta_emu_train = X_train.copy()
    x_emu_train = np.array([[0]]) 
    
    f_emu_train = Y_train[:, output_dim_idx].copy().reshape(1, -1)
    
    # Normalize training data for Surmise as well
    f_mean = np.mean(f_emu_train)
    f_std = np.std(f_emu_train)
    if f_std > 0:
        f_emu_train_normalized = (f_emu_train - f_mean) / f_std
    else:
        f_emu_train_normalized = f_emu_train
        f_std = 1.0
    
    emu = emulator(
        x=x_emu_train, 
        theta=theta_emu_train, 
        f=f_emu_train_normalized, 
        method='PCGP', 
        options={'epsilon': 0}
    )
    start_train = time.time()
    emu.fit()
    end_train = time.time()
    train_time = end_train - start_train
    
    start_pred = time.time()
    pred = emu.predict(x=x_emu_train, theta=X_test) 

    Y_pred_mean_norm = pred.mean().flatten()
    Y_pred_std_norm = np.sqrt(pred.var()).flatten()

    end_pred = time.time()
    pred_time = end_pred - start_pred
    
    # Denormalize predictions
    Y_pred_mean = Y_pred_mean_norm * f_std + f_mean
    Y_pred_std = Y_pred_std_norm * f_std
    
    Y_pred_mean = Y_pred_mean.reshape(-1, 1)
    Y_pred_std = Y_pred_std.reshape(-1, 1)
    
    return Y_pred_mean, Y_pred_std, emu, train_time, pred_time