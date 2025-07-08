import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_fit():
    # generate test data
    input_dim = 3
    output_dim = 5
    n_components = 3
    X_train, Y_train, X_test, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    fitted_model = pcgp.fit(X_train, Y_train, ranges)
    print('fdfdfdssvs')
        
    rho_updated = fitted_model.rho is not None and fitted_model.rho.shape == (n_components, input_dim)
    lambda_w_updated = fitted_model.lambda_w is not None and len(fitted_model.lambda_w) == n_components
    noise_var_updated = fitted_model.noise_var is not None
        
    print("\n=== Fit Tests ===")
    print("Hyperparameters (rho):", "PASS" if rho_updated else "FAIL")
    print("Hyperparameters (lambda_w):", "PASS" if lambda_w_updated else "FAIL")
    print("Hyperparameters (noise_var):", "PASS" if noise_var_updated else "FAIL")
        
    return fitted_model

test_fit()