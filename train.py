from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress missing lib msg

import tensorflow as tf


print("TensorFlow version: {}".format(tf.__version__))
print(tf.config.list_physical_devices('GPU'))