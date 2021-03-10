import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
# sign A
input_data = np.array([[[0.58767754, 0.7381983, -1.7575442E-4],
                       [0.44619673, 0.71034455, -0.02160635],
                       [0.34542662, 0.64283997, -0.010356386],
                       [0.3141297, 0.57854307, -0.013939923],
                       [0.3046583, 0.5238068, -0.0152247315],
                       [0.41976416, 0.5848259, 0.07325372],
                       [0.3686071, 0.54374695, -0.038802177],
                       [0.3896383, 0.59538364, -0.12469772],
                       [0.41619164, 0.6391279, -0.15086864],
                       [0.49366796, 0.57792526, 0.05010624],
                       [0.44325283, 0.54389817, -0.08719535],
                       [0.46695474, 0.6103691, -0.15396108],
                       [0.49665642, 0.6603435, -0.15437144],
                       [0.5726923, 0.5761606, 0.0053979903],
                       [0.5248306, 0.54829395, -0.12275957],
                       [0.53752255, 0.61629957, -0.15667194],
                       [0.5557855, 0.66831684, -0.12990193],
                       [0.6612453, 0.5792883, -0.03993318],
                       [0.61076474, 0.5539133, -0.13030052],
                       [0.6011265, 0.60239273, -0.14864293],
                       [0.6097902, 0.6441627, -0.117933035]]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
