"""
Creators: João Farias and Júlio Freire (JF2)
Date: 23 July 2022
Create API
"""

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import Activation

from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import numpy as np


class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class Normalize():
    def __init__(self, value=255):
        self.value = value

    def fit(self, X):
        return self

    def transform(self, X):
        return X/self.value


class NumericalTransformer(BaseEstimator, TransformerMixin):
    # normalize = False: no scaler
    # normalize = True: normalize RBG values
    def __init__(self, normalize=True, image_height=30, image_width=30):
        self.normalize = normalize
        self.image_height = image_height
        self.image_width = image_width
        self.scaler = None

    def fit(self, X, y=None):
        if self.normalize:
          self.scaler = Normalize(255)
          self.scaler.fit(X)
        return self

    # transforming numerical features
    def transform(self, X, y=None):
            X_copy = []

            for img in X:
                image = Image.fromarray(img)
                image = image.resize((self.image_height,self.image_width))
                img = np.array(image)
                X_copy.append(img)

            X_copy = np.array(X_copy)

            if self.normalize:
                X_copy = self.scaler.transform(X_copy)

            return X_copy