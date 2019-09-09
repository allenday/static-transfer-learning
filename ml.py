import json
import os
import uuid
import logging
import datetime
import settings

RANDOM_SEED = 1234

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
import random

random.seed(RANDOM_SEED)

import numpy as np

np.random.seed(RANDOM_SEED)
np.set_printoptions(precision=4)

import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.keras.backend.set_session(sess)
tf.set_random_seed(RANDOM_SEED)

from tensorflow.python.keras.saving import model_from_json
from datamanager import DataManager


class ModelNotFound(BaseException):
    pass


class ModelNotLoaded(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ML(DataManager):
    model = None

    def __get_optimizer(self):
        return 'Adam'

    def save_model(self, model, class_indices, file_name):
        """
        Save model to local file
        """

        if not model:
            raise ModelNotLoaded('Please load model user "load_model" method')

        model_path = self.get_model_path(file_name)
        model_path_weights = os.path.join(model_path, file_name)
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        model.save_weights(model_path_weights, save_format='tf')

        with open(model_path_json, "w") as json_file:
            json_file.write(model.to_json(sort_keys=True))

        with open(model_class_indices, "w") as json_file:
            json_file.write(json.dumps(list(class_indices.keys()), sort_keys=True))

        logging.info('Model saved into {model_path}'.format(model_path=model_path))
        return model_path

    def load_model(self, file_name=settings.DEFAULT_MODEL_FILENAME):
        """
        Load model from local file
        """
        model_path = self.get_model_path(file_name)
        model_path_weights = os.path.join(model_path, file_name)
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        if not os.path.exists(model_path) or not os.path.exists(model_path_json):
            raise ModelNotFound('Model with path {model_path} is invalid'.format(model_path=model_path))

        with open(model_path_json, "r") as json_file:
            model = model_from_json(json_file.read())

        model.load_weights(model_path_weights)

        with open(model_class_indices, "r") as json_file:
            class_indices = json.load(json_file)

        # Compile model for use optimizers
        model.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', 'accuracy'])

        return model, class_indices

    def __get_image_data_generator(self):
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    def __get_callbacks(self):
        # Define the Keras TensorBoard callback.
        callbacks = []
        if settings.TENSORBOARD_LOGS_ENABLED:
            self.makedirs(self.LOG_DIR)
            logdir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            callbacks.append(tensorboard_callback)
        return callbacks

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        self.makedirs([self.MODELS_DIR])
        # self.cleanup([self.TRAIN_DIR, self.VALIDATE_DIR])
        train_size, validate_size = await self.download_train_data(csv_url)

        train_generator = self.__get_image_data_generator().flow_from_directory(
            self.TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=RANDOM_SEED,
            shuffle=True,
            class_mode='categorical')

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=RANDOM_SEED,
            shuffle=True,
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                                input_shape=train_generator.image_shape,
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                    seed=RANDOM_SEED)))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                    seed=RANDOM_SEED)))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64, activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=RANDOM_SEED)))
        model.add(tf.keras.layers.Dense(units=classes_count, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=RANDOM_SEED)))

        model.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        steps_per_epoch = round(train_size) // settings.BATCH_SIZE
        validation_steps = round(validate_size) // settings.BATCH_SIZE

        model.fit_generator(train_generator,
                            epochs=settings.EPOCHS,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            shuffle=False,
                            max_queue_size=1,
                            callbacks=self.__get_callbacks())

        model.summary()

        logging.info('Classes: {classes}'.format(classes="; ".join(
            ['%s:%s' % (i, train_generator.class_indices[i]) for i in train_generator.class_indices.keys()])))

        loss, acc = model.evaluate(validation_generator)
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        return self.save_model(model, train_generator.class_indices, model_uri)

    async def inference(self, image_url, model_filename, output_uri):
        self.makedirs([self.TMP_DIR])

        model, class_indices = self.load_model(model_filename)

        image_tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        await self.download_file({
            "url": image_url,
            "file_path": image_tmp_path
        })

        img = tf.keras.preprocessing.image.load_img(image_tmp_path,
                                                    target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.reshape(img, [settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3])
        img = np.expand_dims(img, axis=0)

        os.remove(image_tmp_path)

        result = {}
        for idx, res in enumerate(list(model.predict(img)[0])):
            result[class_indices[idx]] = res

        print(result)

        return result

    async def evaluate(self, model_filename=settings.DEFAULT_MODEL_FILENAME):
        model, class_indices = self.load_model(model_filename)

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        loss, categorical_accuracy, acc = model.evaluate(validation_generator, use_multiprocessing=True)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
