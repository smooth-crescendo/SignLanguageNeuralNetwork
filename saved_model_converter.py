import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('sign_model')
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)


converter = tf.lite.TFLiteConverter.from_saved_model('sign_model')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
