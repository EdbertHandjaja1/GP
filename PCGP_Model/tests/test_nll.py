import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_nll():
    # generate test data
    input_dim = 3
    output_dim = 5
    n_components = 3
    n_train = 10 
    X_train, Y_train, _, _, ranges, _ = generate_test_data(
        n_train=n_train,
        input_dim=input_dim, 
        output_dim=output_dim
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )

    # test nll
    pcgp.X_train_std = pcgp.standardize_inputs(X_train, ranges)
    pcgp.Y_train_std = pcgp._standardize_output(Y_train)

    _, phi_basis_np = pcgp.compute_principal_components(pcgp.Y_train_std)
    pcgp.phi_basis = phi_basis_np 

    pcgp.n_components = phi_basis_np.shape[1] 
    
    rho_flattened_init = np.random.rand(pcgp.n_components * pcgp.input_dim) * 0.5 + 0.1
    
    lambda_w_init = np.random.rand(pcgp.n_components) * 0.5 + 0.5
    
    noise_var_init = 1e-4

    # results
    nll_value, k = pcgp._negative_log_marginal_likelihood(
        rho_flattened_init,
        lambda_w_init,
        noise_var_init
    )
    nll_pass = isinstance(nll_value, float) and not np.isnan(nll_value) and not np.isinf(nll_value)
    expected_k_shape = pcgp.X_train_std.shape[0] * pcgp.n_components
    k_shape_pass = (k.shape == (expected_k_shape, expected_k_shape))

    print("=== Negative Log Likelihood Test ===")
    print(f"K (nq x nq) shape:", "PASS" if k_shape_pass else "FAIL")
    print(f"NLL calculation:", "PASS" if nll_pass else "FAIL", ": ", nll_value)

test_nll()