import numpy as np
import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)

from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from classifier.predict import Prediction
from load.load_data import Loading
from app.models.models import Classifier
from preprocess.custom_transformers import transformations_pipeline
from preprocess.preprocess import Preprocessing
import requests
from train.train import training
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

app = FastAPI()

def predict(input):
    url3 = "http://app.docker:8000/predict"

    response = requests.post(url3, json=input)
    response = response.text

    return response

@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Front-end is all ready to go!")
    return 'Front-end is all ready to go!'

@app.post("/predict")
def classify(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict(payload)
    logger.debug(f"Prediction: {response}")
    return {"response": response}

@app.post("/predict2")
def classify(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict(payload)
    logger.debug(f"Prediction: {response}")
    return {"response": response}


@app.get("/healthcheck")
async def v1_healhcheck():
    url3 = "http://app.docker:8000/"

    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Checking health: {response}")

    return response