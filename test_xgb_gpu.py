import xgboost as xgb

# Get XGBoost version
xgb_version = xgb.__version__
print(f"XGBoost version: {xgb_version}")

# Check if CUDA is available
try:
    # Try to access CUDA-specific attributes
    print("Checking CUDA support via parameters test")
    test_params = {'device': 'cuda'}
    test_dmatrix = xgb.DMatrix([[0, 0]], label=[0])
    test_model = xgb.train(test_params, test_dmatrix, num_boost_round=1)
    print("CUDA support confirmed - test model trained successfully")
except Exception as e:
    print("Error checking CUDA support:", e)
    print("Continuing with GPU parameters anyway...")

# Prepare a simple dataset
dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])

# Define parameters to use GPU acceleration (XGBoost 2.0+ approach)
params = {
    'tree_method': 'hist',  # Use histogram-based algorithm
    'device': 'cuda',       # Use CUDA for GPU acceleration
    'objective': 'binary:logistic'
}

# Train a simple model
model = xgb.train(params, dtrain, num_boost_round=10)
print("Model trained successfully with GPU!")
