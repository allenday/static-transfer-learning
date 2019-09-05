from __future__ import absolute_import, division, print_function, unicode_literals

import os
import uuid
import logging
import datetime
import settings
import numpy as np
import random as rn
from datamanager import DataManager

RANDOM_SEED = 1

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(RANDOM_SEED)

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(RANDOM_SEED)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(RANDOM_SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


class ModelNotFound(BaseException):
    pass


class ModelNotLoaded(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ML(DataManager):
    IMG_SHAPE = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)
    model = None

    def __get_optimizer(self):
        return Adam()

    def save_model(self, file_name):
        """
        Save model to local file
        """

        if not self.model:
            raise ModelNotLoaded('Please load model user "load_model" method')

        model_path = self.get_model_path(file_name)
        model_path_weights = model_path + '.h5'
        model_path_json = model_path + '.json'

        self.model.save_weights(model_path_weights)

        with open(model_path_json, "w") as json_file:
            json_file.write(self.model.to_json())

        logging.info('Model saved into {model_path}'.format(model_path=model_path))
        return model_path

    def load_model(self, file_name=settings.DEFAULT_MODEL_FILENAME):
        """
        Load model from local file
        """
        model_path = self.get_model_path(file_name)
        model_path_weights = os.path.join(model_path, file_name)
        model_path_json = os.path.join(model_path, 'model.json')

        if not os.path.exists(model_path) or not os.path.exists(model_path_json):
            raise ModelNotFound('Model with path {model_path} is invalid'.format(model_path=model_path))

        with open(model_path_json, "r") as json_file:
            model = model_from_json(json_file.read())

        model.load_weights(model_path_weights)

        # Compile model for use optimizers
        model.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', 'accuracy'])

        return model

    def __get_image_data_generator(self):
        return ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        self.makedirs([self.MODELS_DIR])
        self.cleanup([self.TRAIN_DIR, self.VALIDATE_DIR])
        train_size, validate_size = await self.download_train_data(csv_url)

        train_generator = self.__get_image_data_generator().flow_from_directory(
            self.TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=RANDOM_SEED,
            class_mode='categorical')

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=RANDOM_SEED,
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        self.model = Sequential()
        self.model.add(keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                                  input_shape=self.IMG_SHAPE,
                                                  kernel_initializer=keras.initializers.glorot_uniform(
                                                      seed=RANDOM_SEED)))
        # self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu',
                                                  kernel_initializer=keras.initializers.glorot_uniform(
                                                      seed=RANDOM_SEED)))
        # self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=64, activation='relu',
                                          kernel_initializer=keras.initializers.glorot_uniform(seed=RANDOM_SEED)))
        self.model.add(keras.layers.Dense(units=classes_count, activation='softmax',
                                          kernel_initializer=keras.initializers.glorot_uniform(seed=RANDOM_SEED)))

        self.model.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])

        steps_per_epoch = round(train_size) // settings.BATCH_SIZE
        validation_steps = round(validate_size) // settings.BATCH_SIZE

        # Define the Keras TensorBoard callback.
        callbacks = []
        if settings.TENSORBOARD_LOGS_ENABLED:
            self.makedirs(self.LOG_DIR)
            logdir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=logdir)
            callbacks.append(tensorboard_callback)

        self.model.fit_generator(train_generator,
                                 epochs=settings.EPOCHS,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps,
                                 max_queue_size=1,
                                 shuffle=False,
                                 callbacks=callbacks)

        self.model.summary()

        logging.info('Classes: {classes}'.format(classes="; ".join(
            ['%s:%s' % (i, train_generator.class_indices[i]) for i in train_generator.class_indices.keys()])))

        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator)
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        return self.save_model(model_uri)

    async def inference(self, image_url, model_uri, output_uri):
        self.makedirs([self.TMP_DIR])

        self.model = self.load_model(model_uri)

        image_tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        await self.download_file({
            "url": image_url,
            "file_path": image_tmp_path
        })

        img = load_img(image_tmp_path,
                       target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = img_to_array(img)
        img = np.reshape(img, [settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3])
        img = np.expand_dims(img, axis=0)

        result = self.model.predict_classes(img)

        print(result)

        os.remove(image_tmp_path)

        return result

    async def evaluate(self, model_filename):
        self.model = self.load_model(model_filename)

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator, use_multiprocessing=True)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
