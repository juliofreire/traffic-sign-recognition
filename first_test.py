from PIL import Image
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
import wandb


# im = Image.("plc.png")
# im = Image.fromarray(im)
# im = im.resize((30, 30))
# print(im.shape)
# print(type(im))

#ima.show()
#ima3 = Image.fromarray(ima)

with h5py.File('plc.h5', 'w') as hf:
    # Images
    image = cv2.imread('pls.png')
    image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_CUBIC)
    Xset = hf.create_dataset(name='X', data=image, shape=(30, 30, 3), maxshape=(30, 30, 3),compression="gzip", compression_opts=9)

with h5py.File('plc.h5', 'a') as hf:
    img_array = np.array(Image.open('plc.png'))
    dset = hf.create_dataset('plc.png', data=img_array)

with h5py.File('plc.h5', 'r') as hf:
    print("Keys: %s" % hf.keys())
    a_group_key = list(hf.keys())[0]
    print(type(hf[a_group_key]))
    data = list(hf[a_group_key])
    data_array = np.array(data)

print(type(dset))
print("---")
print(data)
print(type(data))
print("---")
print(data_array)
print(type(data_array))
print(data_array.shape)
im = Image.open('pls.png')
im = im.resize((30,30), Image.ANTIALIAS)
data_array = np.asfarray(im)

dataz = np.expand_dims(data_array, 0)
print(dataz.shape)


import wandb

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

print("---")
print(model.layers)
model_charged = charge_model(model)

def predict(image):
     prediction = model.predict(image)

     max_val = 0
     idx_max = 0

     for i in range(43):
         if prediction[0][i] > max_val:
             max_val = prediction[0][i]
             idx_max = i

     print(max_val, idx_max)

     return prediction, max_val, idx_max

print(predict(dataz))