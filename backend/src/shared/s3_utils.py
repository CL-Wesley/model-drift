"""
S3 utilities for loading data and models from external S3 API
"""
import os
import logging
import pickle
import joblib
import requests
import pandas as pd
from io import BytesIO
from typing import Dict, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# --- Metadata retrieval ---
def get_s3_file_metadata(project_id: str) -> Dict[str, Any]:
    """
    Lists files and models from the external S3 API and returns their metadata.
    Separates files and models based on the folder field.
    """
    file_api = os.getenv("FILES_API_BASE_URL")
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ3ZXNsZXkuc2FtdGhvbWFzQGNpcnJ1c2xhYnMuaW8iLCJ1c2VyX2lkIjo2NSwicm9sZXMiOltdLCJwZXJtaXNzaW9ucyI6W10sImV4cCI6MTc1OTM4NzcxNH0.vzbPCK_rqnKMitgKv8Bse9ngfOwB-hQT920ezcm4W8E"

    if not file_api:
        raise HTTPException(status_code=500, detail="FILES_API_BASE_URL env variable is not set.")
    if not token:
        raise HTTPException(status_code=401, detail="User token is missing or invalid.")

    url = f"{file_api}/Data Drift/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        json_data = resp.json()
        all_items = json_data.get("files", [])

        return {
            "files": [i for i in all_items if i.get("folder") == "files"],
            "models": [i for i in all_items if i.get("folder") == "models"],
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"S3 API connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to S3 API: {str(e)}")
    except Exception as e:
        logger.error(f"S3 API response error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing S3 API response: {str(e)}")

# --- CSV loading ---
def load_s3_csv(url: str) -> pd.DataFrame:
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(BytesIO(resp.content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        logger.info(f"Loaded CSV from {url}: shape={df.shape}")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download CSV file: {str(e)}")
    except pd.errors.ParserError as e:
        logger.error(f"Invalid CSV format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected CSV load error: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading CSV file: {str(e)}")

# --- Model loading ---
def load_s3_model(url: str) -> Any:
    """
    Load a model from S3 URL (supports pickle, joblib, and ONNX formats).
    This tries to return the same object you saved (pipeline or estimator).
    """
    if not url:
        raise HTTPException(status_code=400, detail="Model URL is required")

    # pre-import common ML classes for pickle
    _prime_globals_for_sklearn()

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        content_io = BytesIO(resp.content)
        ext = url.split(".")[-1].lower() if "." in url else ""

        if ext == "onnx":
            return _load_onnx(content_io, url)

        # Try joblib first (it’s faster)
        try:
            model = joblib.load(content_io)
            logger.info(f"Loaded model from {url} via joblib")
        except Exception:
            content_io.seek(0)
            model = pickle.load(content_io)
            logger.info(f"Loaded model from {url} via pickle")

        model = _unwrap_model_dicts(model)
        if not hasattr(model, "predict"):
            raise HTTPException(status_code=400,
                                detail="Loaded model does not expose a 'predict' method. "
                                       "Ensure you saved a full pipeline or wrap preprocessing.")
        return model
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading model from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model file: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected model load error from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model file: {str(e)}")

def _prime_globals_for_sklearn():
    """Pre-import sklearn/xgboost/lightgbm/catboost classes into globals for pickle"""
    try:
        import sklearn
        import sklearn.ensemble, sklearn.linear_model, sklearn.svm, sklearn.tree
        import sklearn.neighbors, sklearn.naive_bayes, sklearn.neural_network
        from sklearn.pipeline import Pipeline
        # Put some common classes into globals
        globals().update({
            "Pipeline": Pipeline,
            "RandomForestClassifier": sklearn.ensemble.RandomForestClassifier,
            "RandomForestRegressor": sklearn.ensemble.RandomForestRegressor,
            "GradientBoostingClassifier": sklearn.ensemble.GradientBoostingClassifier,
            "GradientBoostingRegressor": sklearn.ensemble.GradientBoostingRegressor,
            "LogisticRegression": sklearn.linear_model.LogisticRegression,
            "LinearRegression": sklearn.linear_model.LinearRegression,
            "SVC": sklearn.svm.SVC,
            "SVR": sklearn.svm.SVR,
            "DecisionTreeClassifier": sklearn.tree.DecisionTreeClassifier,
            "DecisionTreeRegressor": sklearn.tree.DecisionTreeRegressor,
        })
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier, XGBRegressor
        globals().update({"XGBClassifier": XGBClassifier, "XGBRegressor": XGBRegressor})
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        globals().update({"LGBMClassifier": LGBMClassifier, "LGBMRegressor": LGBMRegressor})
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        globals().update({"CatBoostClassifier": CatBoostClassifier, "CatBoostRegressor": CatBoostRegressor})
    except ImportError:
        pass

def _load_onnx(content_io: BytesIO, url: str):
    try:
        import onnxruntime as ort
        model = ort.InferenceSession(content_io.getvalue())
        logger.info(f"Loaded ONNX model from {url}")
        return model
    except ImportError:
        raise HTTPException(status_code=500, detail="onnxruntime is required for ONNX models")

def _unwrap_model_dicts(model: Any) -> Any:
    """If the loaded object is a dict, extract the estimator or wrap it."""
    if not isinstance(model, dict):
        return model
    logger.info("Model loaded as dictionary, looking for model object within...")
    # common keys:
    for k in ["model", "estimator", "classifier", "regressor", "pipeline"]:
        if k in model and hasattr(model[k], "predict"):
            logger.info(f"Found model object in dictionary under key: {k}")
            return model[k]

    # pipeline stored as dict of steps
    if "steps" in model and isinstance(model["steps"], list):
        try:
            from sklearn.pipeline import Pipeline
            return Pipeline(model["steps"])
        except Exception:
            pass

    # fallback wrapper
    class DictModelWrapper:
        def __init__(self, d):
            self._d = d
            self.best_estimator_ = d.get("best_estimator_")
        def predict(self, X):
            if self.best_estimator_:
                return self.best_estimator_.predict(X)
            raise HTTPException(status_code=400, detail="Model dict does not contain a usable estimator.")
        def __getattr__(self, name):
            return getattr(self._d, name) if hasattr(self._d, name) else self._d.get(name)
    return DictModelWrapper(model)

# --- Validators ---
def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise HTTPException(status_code=400, detail=f"{name} dataset is empty")
    if df.shape[0] < 2:
        raise HTTPException(status_code=400, detail=f"{name} dataset must have at least 2 rows")
    if df.shape[1] < 1:
        raise HTTPException(status_code=400, detail=f"{name} dataset must have at least 1 column")

def validate_target_column(df: pd.DataFrame, target_column: str, df_name: str) -> None:
    if target_column not in df.columns:
        raise HTTPException(status_code=400,
                            detail=f"Target column '{target_column}' not found in {df_name} dataset. "
                                   f"Available columns: {list(df.columns)}")
