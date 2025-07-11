import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel, generate_test_data

def test_fit():
    # generate test data
    input_dim = 3
    output_dim = 5
    n_components = 3
    X_train, Y_train, _, _, ranges, _ = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    fitted_model = pcgp.fit(X_train, Y_train, ranges)
        
    rho_updated = fitted_model.rho is not None and fitted_model.rho.shape == (n_components, input_dim)
    lambda_w_updated = fitted_model.lambda_w is not None and len(fitted_model.lambda_w) == n_components
    noise_var_updated = fitted_model.noise_var is not None
        
    print("\n=== Fit Tests ===")
    print("Hyperparameters (rho):", "PASS" if rho_updated else "FAIL")
    print("Hyperparameters (lambda_w):", "PASS" if lambda_w_updated else "FAIL")
    print("Hyperparameters (noise_var):", "PASS" if noise_var_updated else "FAIL")
        
    return fitted_model

test_fit()

# Optimization completed in 673 iterations
# Success: True
# Final NLL: 351.012029

# Optimized hyperparameters:
# Component 1:
#   Length scales (ρ): [0.10165914 0.03661507 0.00734365]
#   Precision (λ): 148.4132
# Component 2:
#   Length scales (ρ): [4.53999298e-05 4.53999298e-05 4.53999298e-05]
#   Precision (λ): 1.5218
# Component 3:
#   Length scales (ρ): [3.01743863e-03 8.97942316e-03 4.53999298e-05]
#   Precision (λ): 3.3930
# Noise variance: 0.808792