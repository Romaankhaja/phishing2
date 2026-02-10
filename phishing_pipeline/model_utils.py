import joblib
from .config import *
import sys, asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
 

class MockEstimator:
    def predict(self, X):
        # Return zeros or a default class '0' (Legitimate)
        import numpy as np
        return np.zeros(X.shape[0] if hasattr(X, "shape") else len(X), dtype=int)
    
    def transform(self, X):
        return X

class MockTransformer:
    def transform(self, X):
        return X
    def fit_transform(self, X, y=None):
        return X

def load_models_and_preproc():
    try:
        model_label = joblib.load(MODEL_LABEL_PATH)
        model_source = joblib.load(MODEL_SOURCE_PATH)
        le_label = joblib.load(ENCODER_LABEL_PATH)
        source_classes = joblib.load(SOURCE_CLASSES_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model_label, model_source, le_label, source_classes, feature_columns, scaler, imputer
    except Exception as e:
        print(f"⚠️ Model loading failed ({e}). Using MOCK models for benchmarking/safety.")
        
        # Return mocks
        model_label = MockEstimator()
        model_source = MockEstimator()
        
        # Mock label encoder
        class MockLE:
            def inverse_transform(self, y):
                return ["Suspected"] * len(y)
        le_label = MockLE()
        
        # Mock source classes (list)
        source_classes = ["Unknown"] * 10
        
        # Mock feature columns (needs to be list)
        feature_columns = ["mock_feat"]
        
        scaler = MockTransformer()
        imputer = MockTransformer()
        
        return model_label, model_source, le_label, source_classes, feature_columns, scaler, imputer
