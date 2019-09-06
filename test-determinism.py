import tensorflow as tf
import time
# PREPROCESSING STAGE
import numpy as np

# Use seeds to keep the production of the datasets deterministic
np.random.seed(42)
data = np.random.random((2000, 32))
print(data)

np.random.seed(42)
labels = np.random.random((2000, 10))
print(labels)

training_data = data[:1000]
training_labels = labels[:1000]

validation_data = data[1000:]
validation_labels = labels[1000:]

training_tuple = (training_data, training_labels)
validation_tuple = (validation_data, validation_labels)

training_set = tf.data.Dataset.from_tensor_slices(training_tuple)
validation_set = tf.data.Dataset.from_tensor_slices(validation_tuple)

# TRAINING STAGE
tf.compat.v1.random.set_random_seed(1234)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set a random seed to keep the fitting of the model deterministic
tf.set_random_seed(42)
model.fit(
    training_data, training_labels,
    epochs=10,
    steps_per_epoch=10,
    batch_size=32,
    shuffle=False,
    max_queue_size=1,
    validation_data=(validation_data, validation_labels)
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# model.save_weights("1/model", save_format='tf')
# time.sleep(1)
# model.save_weights("1/model2", save_format='tf')
