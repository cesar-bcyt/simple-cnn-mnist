import numpy as np
import tensorflow as tf
import gzip

def convert_line(line):
    line = line[2:].strip()
    return list(map(lambda value: float(value), line.split(',')))

def load_data(filename):
    raw_data = gzip.open(filename, 'rt').readlines()[1:]
    raw_data = list(map(lambda x: (int(x[0]), convert_line(x)), raw_data))
    labels = [row[0] for row in raw_data]
    data = [row[1] for row in raw_data]
    return (np.array(data).reshape(-1, 28, 28, 1), np.array(labels))

train_data, train_labels = load_data('./mnist_dataset/mnist_train.csv.gz')
test_data, test_labels = load_data('./mnist_dataset/mnist_test.csv.gz')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
model.fit(x=train_data, y=train_labels, epochs=6)
model.evaluate(x=test_data, y=test_labels)
