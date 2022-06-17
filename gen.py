# github.com/jcwml
import sys
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from random import seed
from time import time_ns
from sys import exit
from os.path import isfile
from os import mkdir
from os.path import isdir
from struct import pack

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

# hyperparameters
seed(74035)
samples = 9
model_name = 'keras_model'
audio_path = 'song.raw'
audio_file = 'song.raw'
audio_name = 'song'

# load options
print("\n--Configuration")
argc = len(sys.argv)
if argc >= 2:
    model_name = sys.argv[1]
    print("model_name:", model_name)
if argc >= 3:
    audio_path = sys.argv[2]
    audio_file = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_file)[0]
    print("audio_path:", audio_path)
if argc >= 4:
    samples = int(sys.argv[3])
    print("samples:", samples)

##########################################
#   PREDICT
##########################################
print("\n--Predicting")
st = time_ns()

model = keras.models.load_model(model_name)

pssb = os.stat(audio_path).st_size
print(pssb)
pssb = pssb - (pssb - (int(pssb / samples)*samples)) # cut off any excess
print(pssb)
pss = int(pssb / samples)
print("Prediction Size:", "{:,}".format(pss))

predict_x = []

if isfile(audio_file + ".npy"):
    predict_x = np.load(audio_file + ".npy")
    predict_x = np.reshape(predict_x, [pss, samples])
    print("Loaded numpy array")
else:

    print(".. Loading prediction track")

    lp_x = []
    with open(audio_path, 'rb') as f:
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

    np.save(audio_file + ".npy", predict_x)

print(".. predicting")

if not isdir('outputs'): mkdir('outputs')

f = open("outputs/" + audio_name + ".raw", "wb")
if f:

    p = model.predict(predict_x)
    # print(p)
    print(".. exporting")
    for i in range(len(p)):
        for j in range(samples):
            # print(p[i][j], p[i][j]*255.0, int(p[i][j]*255.0))
            f.write(pack('B', clamp(int(p[i][j]*255.0), -255, 255)))

    f.close()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds\n")