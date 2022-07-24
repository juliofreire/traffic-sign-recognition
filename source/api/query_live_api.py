"""
Creators: João Farias and Júlio Freire (JF2)
Date: 23 July 2022
Create API
"""

from PIL import Image
from predict import *
#import tensorflow as tf
#from tensorflow import keras
import requests
import json

#image = Image.open('pls.png')
#image = read_image_png('pls.png')
#image = preprocess(image)
image = {'file': ('file', open('ple.png', 'rb'), 'multipart/form-data')}

url = "http://127.0.0.1:8000"
response = requests.post(f"{url}/predict", files=image)

print(f"Request: {url}")
r = json.loads(response.text)
if image != 0:
    print("There is an image")
    print(f"It's ok, status: {response.status_code}")
    print(f"ClassId:{r['pred']}, ClassName:{r['classe']}")
else:
    print("There isn't an image")