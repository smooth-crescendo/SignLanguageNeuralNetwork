import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)


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
    return result


def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {0: "letter a", 1: "letter b", 2: "letter c", 3: "letter d", 4: "letter e"}

    points_a = read_data("training_data_a.txt")
    points_b = read_data("training_data_b.txt")
    points_c = read_data("training_data_c.txt")
    points_d = read_data("training_data_d.txt")
    points_e = read_data("training_data_e.txt")

    test_points.append(points_a.pop(0))
    test_points.append(points_a.pop(8))
    test_points.append(points_b.pop(4))
    test_points.append(points_b.pop(8))
    test_points.append(points_c.pop(4))
    test_points.append(points_c.pop(8))
    test_points.append(points_d.pop(4))
    test_points.append(points_e.pop(4))

    test_labels.append(0)
    test_labels.append(0)
    test_labels.append(1)
    test_labels.append(1)
    test_labels.append(2)
    test_labels.append(2)
    test_labels.append(3)
    test_labels.append(4)

    train_points.extend(points_a)
    train_points.extend(points_b)
    train_points.extend(points_c)
    train_points.extend(points_d)
    train_points.extend(points_e)

    train_labels.extend([0] * len(points_a))
    train_labels.extend([1] * len(points_b))
    train_labels.extend([2] * len(points_c))
    train_labels.extend([3] * len(points_d))
    train_labels.extend([4] * len(points_e))

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
NUM_CLASSES = 5
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

outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="signs_probabilities")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

# Save the model
model.save('sign_model')

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:1, ...]
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
