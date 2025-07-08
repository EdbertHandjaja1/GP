import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcgp1 import PrincipalComponentGaussianProcessModel, generate_test_data

def test_predict():
    # generate test data
    input_dim = 3
    output_dim = 5
    n_components = 3
    X_train, Y_train, X_test, Y_test, ranges, _ = generate_test_data(
        input_dim=input_dim, 
        output_dim=output_dim
    )

    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    fitted_model = pcgp.fit(X_train, Y_train, ranges)
    
    predictions = fitted_model.predict(X_test, ranges)
    predictions_with_std = fitted_model.predict(X_test, ranges, return_std=True)
    
    pred_shape_ok = predictions.shape == Y_test.shape
    
    print("\n=== Predict Tests ===")
    print("Predictions shape matches:", "PASS" if pred_shape_ok else "FAIL")
    print("Predictions with std returns tuple:", "PASS" if isinstance(predictions_with_std, tuple) else "FAIL")
    print("Mean predictions shape:", "PASS" if predictions_with_std[0].shape == Y_test.shape else "FAIL")
    print("Std predictions shape:", "PASS" if predictions_with_std[1].shape == Y_test.shape else "FAIL")

test_predict()