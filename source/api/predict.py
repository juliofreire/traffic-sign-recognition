from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
import wandb


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image


def read_image_png(image_encoded):
    pil_image = Image.open(image_encoded)
    return pil_image


def preprocess(image: Image.Image):

    image = image.resize((30, 30), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asfarray(image)
    image = image/255
    image = np.expand_dims(image, 0)
    print(image.shape)
    return image


def load_model():
    # setup wandb
    # neural network layers
    lenet5 = Sequential()

    # testing the effects of batch-normalization (before activation function)
    if 2 != 1:
        lenet5.add(Conv2D(6, (5, 5), strides=1, kernel_initializer='glorot_uniform', activation='tanh',
                          input_shape=(30, 30, 3), padding='same'))
    else:
        lenet5.add(
            Conv2D(6, (5, 5), strides=1, kernel_initializer='glorot_uniform', input_shape=(30, 30, 3), padding='same'))
        lenet5.add(BatchNormalization())
        lenet5.add(Activation('tanh'))

    # testing the effects of batch-normalization (after activation function)
    if 2 == 2:
        lenet5.add(BatchNormalization())

    lenet5.add(AveragePooling2D())
    lenet5.add(Conv2D(16, (5, 5), strides=1, activation='tanh', padding='valid'))

    # testing the addition of dropout layers
    if False:
        lenet5.add(Dropout(rate=0.5))

    lenet5.add(AveragePooling2D())  # S4
    lenet5.add(Flatten())  # Flatten
    lenet5.add(Dense(120, activation='tanh'))  # C5

    # testing the addition of dropout layers
    if False:
        lenet5.add(Dropout(rate=0.5))

    lenet5.add(Dense(84, activation='tanh'))  # F6

    # testing the addition of dropout layers
    if False:
        lenet5.add(Dropout(rate=0.25))
    lenet5.add(Dense(43, activation='softmax'))  # Output layer
    return lenet5

def charge_model(model):
    best_model = wandb.restore('model-best.h5', run_path="traffic_sign_recognition/34xmkdgt")
    model.load_weights(best_model.name)
    return model

model = load_model()
model_charged = charge_model(model)

def predict(image: np.ndarray):
    prediction = model.predict(image)

    max_val = 0
    idx_max = 0

    for i in range(43):
        if prediction[0][i] > max_val:
            max_val = prediction[0][i]
            idx_max = i

    classes = {0: 'Speed limit (20km/h)',
               1: 'Speed limit (30km/h)',
               2: 'Speed limit (50km/h)',
               3: 'Speed limit (60km/h)',
               4: 'Speed limit (70km/h)',
               5: 'Speed limit (80km/h)',
               6: 'End of speed limit (80km/h)',
               7: 'Speed limit (100km/h)',
               8: 'Speed limit (120km/h)',
               9: 'No passing',
               10: 'No passing veh over 3.5 tons',
               11: 'Right-of-way at intersection',
               12: 'Priority road',
               13: 'Yield',
               14: 'Stop',
               15: 'No vehicles',
               16: 'Veh > 3.5 tons prohibited',
               17: 'No entry',
               18: 'General caution',
               19: 'Dangerous curve left',
               20: 'Dangerous curve right',
               21: 'Double curve',
               22: 'Bumpy road',
               23: 'Slippery road',
               24: 'Road narrows on the right',
               25: 'Road work',
               26: 'Traffic signals',
               27: 'Pedestrians',
               28: 'Children crossing',
               29: 'Bicycles crossing',
               30: 'Beware of ice/snow',
               31: 'Wild animals crossing',
               32: 'End speed + passing limits',
               33: 'Turn right ahead',
               34: 'Turn left ahead',
               35: 'Ahead only',
               36: 'Go straight or right',
               37: 'Go straight or left',
               38: 'Keep right',
               39: 'Keep left',
               40: 'Roundabout mandatory',
               41: 'End of no passing',
               42: 'End no passing veh > 3.5 tons'}

    print(max_val, idx_max, classes[idx_max])

    return idx_max, classes[idx_max]

#return prediction