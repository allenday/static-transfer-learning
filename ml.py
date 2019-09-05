from __future__ import absolute_import, division, print_function, unicode_literals

import os
import uuid
import logging
import datetime
import settings
import numpy as np
import random as rn
import tensorflow as tf
from datamanager import DataManager

os.environ['THEANO_FLAGS'] = 'dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic'
os.environ['PYTHONHASHSEED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, device_count={'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.keras.backend.set_session(sess)


class ModelNotFound(BaseException):
    pass


class ModelNotLoaded(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ML(DataManager):
    IMG_SHAPE = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)
    model = None

    def save_model(self, file_name):
        """
        Save model to local file
        """

        if not self.model:
            raise ModelNotLoaded('Please load model user "load_model" method')

        model_path = self.get_model_path(file_name)
        model_path_weights = os.path.join(model_path, file_name)
        model_path_json = os.path.join(model_path, 'model.json')

        self.model.save_weights(model_path_weights, save_format='tf')

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
            model = tf.keras.models.model_from_json(json_file.read())

        model.load_weights(model_path_weights)

        # Compile model for use optimizers
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', 'accuracy'])

        return model

    def __get_image_data_generator(self):
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        self.makedirs([self.LOG_DIR, self.MODELS_DIR])
        self.cleanup([self.TRAIN_DIR, self.VALIDATE_DIR])
        train_size, validate_size = await self.download_train_data(csv_url)

        np.random.seed(1)
        rn.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        train_generator = self.__get_image_data_generator().flow_from_directory(
            self.TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            seed=1,
            batch_size=settings.BATCH_SIZE,
            interpolation='lanczos',
            class_mode='categorical')

        np.random.seed(1)
        rn.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        validation_generator = self.__get_image_data_generator().flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            seed=1,
            batch_size=settings.BATCH_SIZE,
            interpolation='lanczos',
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        np.random.seed(1)
        rn.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                          input_shape=self.IMG_SHAPE))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=classes_count, activation='softmax'))

        tf.compat.v1.random.set_random_seed(1)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])

        steps_per_epoch = round(train_size) // settings.BATCH_SIZE
        validation_steps = round(validate_size) // settings.BATCH_SIZE

        # Define the Keras TensorBoard callback.
        callbacks = []
        if settings.TENSORBOARD_LOGS_ENABLED:
            logdir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            callbacks.append(tensorboard_callback)

        np.random.seed(1)
        rn.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        self.model.fit_generator(train_generator,
                                 epochs=settings.EPOCHS,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks)

        self.model.summary()

        logging.info('Classes: {classes}'.format(classes="; ".join(
            ['%s:%s' % (i, train_generator.class_indices[i]) for i in train_generator.class_indices.keys()])))

        np.random.seed(1)
        rn.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator, use_multiprocessing=True)
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

        img = tf.keras.preprocessing.image.load_img(image_tmp_path,
                                                    target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
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
            seed=1,
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator, use_multiprocessing=True)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
