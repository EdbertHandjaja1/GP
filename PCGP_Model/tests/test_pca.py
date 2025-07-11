import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel, generate_test_data
import tensorflow as tf

def test_pca():
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

    # test pca
    Y_train_std = pcgp._standardize_output(Y_train)
    weights, phi_basis = pcgp.compute_principal_components(Y_train_std)

    # results
    expected_weights_shape = (Y_train_std.shape[0], n_components)
    weights_shape_pass = (weights.shape == expected_weights_shape)
    expected_phi_basis_shape = (output_dim, n_components)
    phi_basis_shape_pass = (phi_basis.shape == expected_phi_basis_shape)

    print("=== PCA Tests ===")
    print(f"Weights shape check:", "PASS" if weights_shape_pass else "FAIL")
    print(f"Phi basis shape check:", "PASS" if phi_basis_shape_pass else "FAIL")

test_pca()