# github.com/jcwml
import sys
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from random import seed
from time import time_ns
from sys import exit
from os.path import isfile
from os import mkdir
from os.path import isdir
from struct import pack

# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices())
# exit();

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# https://stackoverflow.com/questions/5996881/how-to-limit-a-number-to-be-within-a-specified-range-python
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# hyperparameters
seed(74035)
model_name = 'keras_model'
optimiser = 'adam'
epoches = 64
activator = 'gelu'
layers = 6
layer_units = 32
batches = 999
samples = 9

# load options
print("\n--Configuration")
argc = len(sys.argv)
if argc >= 2:
    layers = int(sys.argv[1])
    print("layers:", layers)
if argc >= 3:
    layer_units = int(sys.argv[2])
    print("layer_units:", layer_units)
if argc >= 4:
    batches = int(sys.argv[3])
    print("batches:", batches)
if argc >= 5:
    activator = sys.argv[4]
    print("activator:", activator)
if argc >= 6:
    optimiser = sys.argv[5]
    print("optimiser:", optimiser)
if argc >= 7 and sys.argv[6] == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU_ONLY: 1")
if argc >= 8:
    samples = int(sys.argv[7])
    print("samples:", samples)
if argc >= 9:
    epoches = int(sys.argv[8])
    print("epoches:", epoches)

# make sure save dir exists
if not isdir('models'): mkdir('models')
model_name = 'models/' + activator + '_' + optimiser + '_' + str(layers) + '_' + str(layer_units) + '_' + str(batches) + '_' + str(samples) + '_' + str(epoches)

##########################################
#   LOAD DATASET
##########################################
print("\n--Loading Dataset")
st = time_ns()

tssb = os.stat("train_x.raw").st_size
print(tssb)
tssb = tssb - (tssb - (int(tssb / samples)*samples)) # cut off any excess
print(tssb)
tss = int(tssb / samples)
print("Dataset Size:", "{:,}".format(tss))

train_x = []
train_y = []

if isfile("train_x.npy"):
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    train_x = np.reshape(train_x, [tss, samples])
    train_y = np.reshape(train_y, [tss, samples])
    print("Loaded shuffled numpy arrays")
else:

    print(".. loading raw files")

    load_x = []
    with open("train_x.raw", 'rb') as f:
        load_x = np.fromfile(f, dtype=np.ubyte)

    load_y = []
    with open("train_y.raw", 'rb') as f:
        load_y = np.fromfile(f, dtype=np.ubyte)

    print(".. normalising arrays")

    train_x = np.empty([tssb, 1], float)
    train_y = np.empty([tssb, 1], float)

    for i in range(tssb):
        train_x[i] = float(load_x[i]) / 255
        train_y[i] = float(load_y[i]) / 255
        sys.stdout.write("\r{:.2f}".format((float(i)/tssb)*100))
        sys.stdout.flush()

    print("\n.. reshaping arrays")

    train_x = np.reshape(train_x, [tss, samples])
    train_y = np.reshape(train_y, [tss, samples])

    print(".. shuffling arrays")

    shuffle_in_unison(train_x, train_y)

    print(".. saving arrays")

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################
print("\n--Training Model")

if isdir(model_name):
    model = keras.models.load_model(model_name)
    print(model_name)
    print("Loaded existing model")
else:
    # construct neural network
    model = Sequential()

    model.add(Dense(layer_units, activation=activator, input_dim=samples))

    for x in range(layers):
        # model.add(Dropout(.1))
        model.add(Dense(layer_units, activation=activator))
        # model.add(BatchNormalization())

    # model.add(Dropout(.3))
    model.add(Dense(samples))

    # output summary
    model.summary()

    if optimiser == 'adam':
        optim = keras.optimizers.Adam(learning_rate=0.001)
    elif optimiser == 'sgd':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*samples, decay_rate=0.1)
        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*samples, decay_rate=0.01)
        optim = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False)
    elif optimiser == 'momentum':
        optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    elif optimiser == 'nesterov':
        optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    elif optimiser == 'nadam':
        optim = keras.optimizers.Nadam(learning_rate=0.001)
    elif optimiser == 'adagrad':
        optim = keras.optimizers.Adagrad(learning_rate=0.001)
    elif optimiser == 'rmsprop':
        optim = keras.optimizers.RMSprop(learning_rate=0.001)
    elif optimiser == 'adadelta':
        optim = keras.optimizers.Adadelta(learning_rate=0.001)
    elif optimiser == 'adamax':
        optim = keras.optimizers.Adamax(learning_rate=0.001)
    elif optimiser == 'ftrl':
        optim = keras.optimizers.Ftrl(learning_rate=0.001)

    model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

    # train network
    history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches)
    model_name = model_name + "_" + "a{:.2f}".format(history.history['accuracy'][-1])
    timetaken = (time_ns()-st)/1e+9
    print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################
print("\n--Exporting Model")
st = time_ns()

# save keras model
model.save(model_name)

pssb = os.stat("song.raw").st_size
print(pssb)
pssb = pssb - (pssb - (int(pssb / samples)*samples)) # cut off any excess
print(pssb)
pss = int(pssb / samples)
print("Prediction Size:", "{:,}".format(pss))

predict_x = []
if isfile("predict_x.npy"):
    predict_x = np.load("predict_x.npy")
    predict_x = np.reshape(predict_x, [pss, samples])
    print("Loaded numpy array")
else:

    print(".. Loading prediction track")

    lp_x = []
    with open("song.raw", 'rb') as f:
        lp_x = np.fromfile(f, dtype=np.ubyte)

    print(".. normalising prediction track")

    predict_x = np.empty([pssb, 1], float)

    for i in range(pssb):
        predict_x[i] = float(lp_x[i]) / 255
        sys.stdout.write("\r{:.2f}".format((float(i)/pssb)*100))
        sys.stdout.flush()

    print("\n.. reshaping prediction track")

    predict_x = np.reshape(predict_x, [pss, samples])

    print(".. saving prediction track")

    np.save("predict_x.npy", predict_x)

print(".. predicting and exporting")

f = open(model_name + ".raw", "wb")
if f:

    p = model.predict(predict_x)
    for i in range(len(p)):
        for j in range(samples):
            f.write(pack('B', clamp(int(p[i][j]*255.0), -255, 255)))

    f.close()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds\n")
