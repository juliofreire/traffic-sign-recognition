"""
Creators: João Farias and Júlio Freire (JF2)
Date: 23 July 2022
Create API
"""

# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import FastAPI, File, UploadFile
import io
from tensorflow import keras
import h5py
import numpy as np
#import cv2
from PIL import Image
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, NumericalTransformer
from source.api.predict import *

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "traffic_sign_recognition/model:latest"

# initiate the wandb project
run = wandb.init(project="traffic_sign_recognition", job_type="api")

# create the api
app = FastAPI()

# Declare request example data using pydantic
# a image in our dataset has the following attributes

# PRENCHER!!!!
class Image(BaseModel):
    filename: str
    list: list
    class Config:
        scheme_exta = {
            "example": {
                "filename": "plc",
                "list": [2, 3]
            }
        }


# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello Welcome to Traffic Sign Recognition </strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired through the second stage of ML course about the Deploying a Neural Network. """\
        """a classification model.</p></span>"""\
    """<p><span style="font-size:20px">For this step, we brought a Convolutional Neural Network with a LeNet-5 architecture and to predict an image """\
        """is necessary acess this link: """\
        """<a href="https://traffic-sign-reco.herokuapp.com/docs"> predict</a> and execute one try.</span></p>"""\
    """<p><span style="font-size:20px"> Our dataset was taken from: """\
        """<a href="https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"> Kaggle: GTSRB - German Traffic Sign Recognition Benchmark</a>.</span></p>
    """


@app.post('/predict')
async def prediction_image(file: bytes = File(...)):
    image = read_image(file)
    image = preprocess(image)
    predictions, classe = predict(image)

    return {"pred": predictions, "classe": classe}

