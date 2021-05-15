import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from math import sqrt

tf.random.set_seed(1234)

NUM_SIGNS = 12


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def get_config(self):
        return {'l2reg': float(self.l2reg), 'num_features': int(self.num_features)}

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def read_data(filename):
    result = []
    with open(filename, 'r') as file:
        hand = []
        for line in file.readlines():
            line = line.strip()
            point = line.split(",")
            for i in range(3):
                point[i] = float(point[i])
            hand.append(point)
            if len(hand) == 21:
                result.append(hand)
                hand = []

    result = align_axis(result, 0)
    result = align_axis(result, 1)
    result = align_axis(result, 2)

    for hand in result:
        normalize(hand)

    return result


def normalize(hand):
    max_axis_value = -100
    for point in hand:
        for axis in point:
            if (axis > max_axis_value):
                max_axis_value = axis
    for point_index in range(len(hand)):
        hand[point_index][0] /= max_axis_value
        hand[point_index][1] /= max_axis_value
        hand[point_index][2] /= max_axis_value


def align_axis(data, ax):
    for j in range(len(data)):
        min = 100
        for i in range(21):
            v = data[j][i][ax]
            if v < min:
                min = v
        for i in range(21):
            data[j][i][ax] -= min
    return data


def parse_dataset(num_points):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    class_map = {}

    for i in range(NUM_SIGNS):
        letter = chr(ord('a') + i)
        class_map.setdefault(i, "letter " + letter)

        points = read_data("training_data/training_data_"+letter+".txt")

        test_points.append(points.pop(4))
        test_labels.append(i)

        train_points.extend(points)

        train_labels.extend([i] * len(points))

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


NUM_POINTS = 21
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)

inputs = keras.Input(shape=(NUM_POINTS, 3), name="hand_landmarks")

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_SIGNS, activation="softmax", name="signs_probabilities")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
# model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=15, validation_data=test_dataset)

# Save the model
model.save('sign_model')

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()
