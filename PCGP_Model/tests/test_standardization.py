import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_standardization():
    # generate test data
    input_dim = 3
    output_dim = 5
    X_train, Y_train, _, _, ranges, _ = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim
    )
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=3,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    # test standardization
    X_std = pcgp.standardize_inputs(X_train, ranges)
    Y_std = pcgp._standardize_output(Y_train)
    Y_unstd = pcgp._unstandardize_output(Y_std)
    
    # results
    input_std_pass = np.all(X_std >= 0) and np.all(X_std <= 1)
    output_recon_pass = np.allclose(Y_train, Y_unstd)
    
    print("=== Standardization Tests ===")
    print("Input standardization check:", "PASS" if input_std_pass else "FAIL")
    print("Output standardization check:", "PASS" if output_recon_pass else "FAIL")

test_standardization()