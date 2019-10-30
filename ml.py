import datetime
import glob
import json
import logging
import os
import shutil
import uuid

import settings
from bgtask import bgt
from storage import storage_factory

os.environ['PYTHONHASHSEED'] = str(settings.RANDOM_SEED)
import random

random.seed(settings.RANDOM_SEED)

import numpy as np

np.random.seed(settings.RANDOM_SEED)
np.set_printoptions(precision=4)

import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(settings.RANDOM_SEED)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.keras.backend.set_session(sess)
tf.set_random_seed(settings.RANDOM_SEED)

from tensorflow.python.keras.saving import model_from_json
from datamanager import DataManager


class ModelNotFound(BaseException):
    pass


class ModelNotLoaded(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ErrorDownloadImage(BaseException):
    pass


class ModelIsLoading(BaseException):
    status = None


class ErrorProcessingImage(BaseException):
    pass


class ML(DataManager):
    models = {}
    READY = 'ready'
    NEW = 'new'
    NOT_FOUND = 'not_found'
    LOADING_START = 'loading_start'
    LOADING_END = 'loading_end'
    IN_PROGRESS = 'in_progress'
    ERROR = 'error'
    DONE = 'done'

    def __get_optimizer(self):
        """
        Tensorflow optimizer getter
        """
        return 'Adam'

    def get_model(self, model_name):
        """
        Tensorflow model getter from memory
        """
        empty_result = {
            'model': None,
            'class_indices': None,
            'error': None,
            'status': self.NOT_FOUND
        }
        return self.models.get(model_name, empty_result)

    def save_model_local(self, model, class_indices, csv_url):
        """
        Save model to local filesystem
        """

        model_name = self.get_model_name(csv_url)
        model_path = self.get_model_path(model_name)
        model_path_weights = os.path.join(model_path, 'model')
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        model.save_weights(model_path_weights, save_format='tf')

        with open(model_path_json, "w") as json_file:
            json_file.write(model.to_json(sort_keys=True))

        with open(model_class_indices, "w") as json_file:
            json_file.write(json.dumps(list(class_indices.keys()), sort_keys=True))

        logging.info('Model saved into {model_path}'.format(model_path=model_path))

        self.models[model_name]['status'] = self.READY

        return model_path

    def load_model_local(self, model_name):
        """
        Load model from local filesystem
        """

        model = self.models.get(model_name)

        if model and model.get('model') and model.get('class_indices') and model.get('status') == self.READY:
            return model

        model_path = self.get_model_path(model_name)
        model_path_weights = os.path.join(model_path, 'model')
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        if not os.path.exists(model_path) or not os.path.exists(model_path_json) or not os.path.exists(
                model_class_indices):
            raise ModelNotFound('Model with path {model_path} is invalid'.format(model_path=model_path))

        model = self.models[model_name] = {}

        with open(model_class_indices, "r") as json_file:
            model['class_indices'] = json.load(json_file)

        with open(model_path_json, "r") as json_file:
            model['model'] = model_from_json(json_file.read())

        model['model'].load_weights(model_path_weights)

        # Compile model for use optimizers
        model['model'].compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                               metrics=['categorical_accuracy', 'accuracy'])

        model['status'] = self.READY

        return model

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

    def __set_model_status(self, model_name, status, error=None):
        if not self.models.get(model_name):
            self.models[model_name] = {}

        self.models[model_name]['status'] = status
        self.models[model_name]['error'] = error

    async def train_local(self, model_name, csv_url):
        """
        Train model by CSV file
        """

        self.makedirs([self.MODELS_DIR])

        self.__set_model_status(model_name, self.IN_PROGRESS)

        train_dir, validate_dir, train_size, validate_size, error = await self.download_train_data(csv_url)
        if error:
            self.__set_model_status(model_name, self.ERROR, error=error)
            return None

        train_generator = self.__get_image_data_generator().flow_from_directory(
            train_dir,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=settings.RANDOM_SEED,
            shuffle=True,
            class_mode='categorical')

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            validate_dir,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=settings.RANDOM_SEED,
            shuffle=True,
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                                input_shape=train_generator.image_shape,
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                    seed=settings.RANDOM_SEED)))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                    seed=settings.RANDOM_SEED)))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64, activation='relu',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(
                                            seed=settings.RANDOM_SEED)))
        model.add(tf.keras.layers.Dense(units=classes_count, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(
                                            seed=settings.RANDOM_SEED)))

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
                            # max_queue_size=1,
                            callbacks=self.__get_callbacks())

        model_path = self.save_model_local(model, train_generator.class_indices, csv_url)

        # Clear TF session after train
        tf.keras.backend.clear_session()

        return model_path

    async def train(self, csv_url, model_uri):
        """
        Train method wrapper for support external storages for model,
        like GCS and IPFS

        TODO: add GCS and IPFS support
        """

        model_path = await self.train_local(self.get_model(self.get_model_name(model_uri)), csv_url)
        try:
            storage_factory.write_data_from_dir(path_to=model_uri, path_from=model_path)
        except Exception as exc:
            error = "Error write data to {model_uri}".format(model_uri=model_uri)
            logging.error('Cant upload data from {model_path}: {error}'.format(model_path=model_path, error=error))
            logging.error(exc)
            return None, None, None, None, error

        return model_path

    async def infer_local(self, image_url, model_name):
        # Prepare
        self.makedirs([self.TMP_DIR])

        image_tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        # Download image
        download_result = await self.download_file({
            "url": image_url,
            "file_path": image_tmp_path
        })
        if not download_result:
            raise ErrorDownloadImage('Error download image by url "{url}"'.format(url=image_url))

        # Init graph & model
        model = self.load_model_local(model_name)

        img = tf.keras.preprocessing.image.load_img(image_tmp_path,
                                                    target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.reshape(img, [settings.IMAGE_SIZE, settings.IMAGE_SIZE, len(model['class_indices'])])
        img = np.expand_dims(img, axis=0)

        result = {}

        graph = tf.get_default_graph()
        with graph.as_default():
            y = model['model'].predict(img)

        for idx, res in enumerate(list(y[0])):
            result[model['class_indices'][idx]] = round(float(res), 2)

        # Clear temporary image file
        os.remove(image_tmp_path)

        return result

    def get_bgtask_by_path(self, path):
        model_name = self.get_model_name(path)

        model = self.get_model(model_name)
        model_status = model.get('status', None)

        result = {
            'model_name': model_name,
            'status': model_status
        }

        return result

    async def prepare_ml_model(self, model_uri):
        bgtask = self.get_bgtask_by_path(model_uri)
        model_path = self.get_model_path(bgtask['model_name'])
        if os.path.exists(model_path):
            bgtask['status'] = self.LOADING_END
            return bgtask

        if bgtask['status'] == self.NOT_FOUND:
            bgtask['status'] = self.LOADING_START
            await bgt.run(self.load_model, [model_uri])

        return bgtask

    async def load_model(self, model_uri):
        model_name = self.get_model_name(model_uri)
        try:
            self.__set_model_status(model_name, self.LOADING_START)
            tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + '/')
            os.mkdir(tmp_path)
            storage_factory.read_data_from_dir(path=model_uri, path_to=tmp_path)
            model_path = self.get_model_path(model_name)
            if not os.path.exists(model_path):
                os.mkdir(model_path)

            for filename in glob.glob(os.path.join(tmp_path, '*.*')):
                shutil.copy(filename, model_path)

            self.__set_model_status(model_name, self.LOADING_END)
        except Exception as exc:
            self.__set_model_status(model_name, self.ERROR)
            logging.error('Can not download model from {model_uri}'.format(model_uri=model_uri))
            logging.error(exc)

    async def infer(self, image_url, model_uri):
        """
        TODO: add GCS and IPFS support
        """

        model_status = await self.prepare_ml_model(model_uri)

        if model_status['status'] == self.READY or model_status['status'] == self.LOADING_END:
            model_path = await self.infer_local(image_url, model_status['model_name'])
            return model_path
        else:
            error = ModelIsLoading()
            error.status = model_status['status']
            raise error
