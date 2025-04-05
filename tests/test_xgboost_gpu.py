import pytest
import xgboost as xgb
import numpy as np

import torch # Used to check for CUDA availability

# Check if CUDA is available and skip test if not
cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA is not available on this system")

def test_xgboost_gpu_support_via_training():
    """
    Tests if XGBoost was compiled with CUDA support by attempting
    to train a simple model using the 'hist' tree_method with device='cuda'.
    This test requires a CUDA-enabled GPU and XGBoost built with CUDA.
    """
    print(f"\n--- XGBoost GPU Support Test ---")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"CUDA available via torch: {cuda_available}")

    # Prepare a simple dataset
    print("Preparing dummy data...")
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    dtrain = xgb.DMatrix(X, label=y)
    print("Dummy data prepared.")

    # Define parameters to use GPU acceleration (modern approach)
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'seed': 42
    }
    print(f"Using parameters: {params}")

    try:
        # Attempt to train the model using the GPU
        print("Attempting to train model on GPU...")
        model = xgb.train(params, dtrain, num_boost_round=5)
        print("Model training successful!")

        # Optional: Make a prediction to ensure the model is usable
        print("Attempting prediction...")
        preds = model.predict(dtrain)
        print(f"Predictions obtained: {preds}")
        assert len(preds) == len(y)
        print("Prediction successful.")

    except xgb.core.XGBoostError as e:
        # This error occurs if XGBoost was not compiled with CUDA support
        # or if there's a CUDA runtime issue.
        print(f"Caught XGBoostError: {e}")
        pytest.fail(f"XGBoost training with device='cuda' failed. Is XGBoost compiled with CUDA support? Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during training or prediction
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
        pytest.fail(f"An unexpected error occurred during the test: {e}")

    print("--- Test Completed Successfully ---")
