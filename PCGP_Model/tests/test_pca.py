import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_pca():
    # generate test data
    input_dim = 3
    output_dim = 5
    X_train, Y_train, X_test, Y_test, ranges, true_func = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=3,
        input_dim=input_dim,
        output_dim=output_dim
    )

    # test_pca
    Y_train_std = pcgp._standardize_output(Y_train)
    K_eta_scores, Phi_basis = pcgp.compute_principal_components(Y_train_std)