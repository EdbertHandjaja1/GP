import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_kernel_matrix():
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
    
    # test kernel
    pcgp.lambda_w = np.array([1.0, 2.0, 0.5])
    pcgp.rho = np.random.rand(n_components, input_dim)
    
    X_train_std = pcgp.standardize_inputs(X_train, ranges)
    X_test_std = pcgp.standardize_inputs(X_test, ranges)
    
    n_train = X_train_std.shape[0]
    n_test = X_test_std.shape[0]

    # results
    K_xx = pcgp._build_kernel_matrix(X_train_std, component_idx=0)
    k_xx_pass = K_xx.shape == (n_train, n_train)
    K_xx_star = pcgp._build_kernel_matrix(X_train_std, X_test_std, component_idx=0)
    k_xx_star_pass = K_xx_star.shape == (n_train, n_test)
    K_x_star_x = pcgp._build_kernel_matrix(X_test_std, X_train_std, component_idx=0)
    k_x_star_x_pass = K_x_star_x.shape == (n_test, n_train)
    K_x_star_x_star = pcgp._build_kernel_matrix(X_test_std, component_idx=0)
    k_x_star_x_star_pass = K_x_star_x_star.shape == (n_test, n_test)

    print("=== Kernel Tests ===")
    print(f"K(x,x): {'PASS' if k_xx_pass else 'FAIL'}")
    print(f"K(x,x*): {'PASS' if k_xx_star_pass else 'FAIL'}")
    print(f"K(x*,x): {'PASS' if k_x_star_x_pass else 'FAIL'}")
    print(f"K(x*,x*): {'PASS' if k_x_star_x_star_pass else 'FAIL'}")

test_kernel_matrix()