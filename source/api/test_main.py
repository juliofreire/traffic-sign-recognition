"""
Creators: João Farias and Júlio Freire (JF2)
Date: 23 July 2022
Create API
"""

from fastapi.testclient import TestClient
import os
import sys
import pathlib
from source.api.main import app

# Instantiate the testing-client with our app
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

#
def test_get_inference():
    image = {'file': ('file', open('ple.png', 'rb'), 'multipart/form-data')}

    url = "http://127.0.0.1:8000"
    r = client.post(f"{url}/predict", files=image)

    assert r.status_code == 200
    assert r.json()['pred'] == 2