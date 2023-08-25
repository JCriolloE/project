import numpy as np
import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from fastapi import HTTPException
from classifier.predict import Prediction
from load.load_data import Loading
from app.models.models import Classifier
from preprocess.custom_transformers import transformations_pipeline
from preprocess.preprocess import Preprocessing
from train.train import training
from utilities import custom_logging
from responses import JSONResponse

logger = custom_logging.CustomLogging(__name__)

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Classifier is all ready to go!")
    return 'Classifier is all ready to go!'

@app.post('/train')
async def training(X, y):
    df = Loading.loadingData("data/")
    df = transformations_pipeline.fit_transform(df)
    X_train, X_test, X_val, y_train, y_test, y_val = Preprocessing.robustScaler(df)
    log_reg = training(X_train, y_train)
    logger.info("Model trained")
    return {"message": "model trained"}

@app.post('/predict')
async def predict(classifier_features: Classifier) -> JSONResponse:
    X = [classifier_features.Time,
        classifier_features.V1,
        classifier_features.V2,
        classifier_features.V3,
        classifier_features.V4,
        classifier_features.V5,
        classifier_features.V6,
        classifier_features.V7,
        classifier_features.V8,
        classifier_features.V9,
        classifier_features.V10,
        classifier_features.V11,
        classifier_features.V12,
        classifier_features.V13,
        classifier_features.V14,
        classifier_features.V15,
        classifier_features.V16,
        classifier_features.V17,
        classifier_features.V18,
        classifier_features.V19,
        classifier_features.V20,
        classifier_features.V21,
        classifier_features.V22,
        classifier_features.V23,
        classifier_features.V24,
        classifier_features.V25,
        classifier_features.V26,
        classifier_features.V27,
        classifier_features.V28,
        classifier_features.Amount]

    try:
        for value in X:
            if not isinstance(value, float):
                logger.error("Value %s must be of type float.", value)
                raise HTTPException(status_code=422, 
                                    detail="Values must be of type float.")
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=422, 
                            detail="Values must be of type float.")

    X = np.array(X)
    X = X.reshape(1, -1)

    predictor = Prediction.predict(X)
    logger.info(f"Resultado predicción: {predictor}")
    return JSONResponse(f"Resultado predicción: {predictor}")