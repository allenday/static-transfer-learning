import json
import os
import shutil
import uuid
import logging
import datetime

import settings
from bgtask import bgt
from helpers import get_sha1_hash
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


class ErrorProcessingImage(BaseException):
    pass


class ModelIsLoading(BaseException):
    status = None


class ML(DataManager):
    models = {}
    NEW = 'new'
    READY = 'ready'
    ERROR = 'error'
    NOT_FOUND = 'not_found'
    IN_PROGRESS = 'in_progress'
    LOADING_START = 'loading_start'
    LOADING_END = 'loading_end'

    def __get_optimizer(self):
        """
        Tensorflow optimizer getter
        """
        return 'Adam'

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

    def __set_model_status(self, model_sha1, status, error=None):
        if not self.models.get(model_sha1):
            self.models[model_sha1] = {}

        self.models[model_sha1]['status'] = status
        self.models[model_sha1]['error'] = error

    def __save_model_local(self, model, class_indices, training_data):
        """
        Save model to local filesystem
        """

        model_sha1 = training_data['model']['sha1']
        model_path = self.get_model_path(model_sha1)
        model_path_weights = os.path.join(model_path, 'model')
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        model.save_weights(model_path_weights, save_format='tf')

        with open(model_path_json, "w") as json_file:
            json_file.write(model.to_json(sort_keys=True))

        with open(model_class_indices, "w") as json_file:
            json_file.write(json.dumps(list(class_indices.keys()), sort_keys=True))

        logging.info('Model saved into {model_path}'.format(model_path=model_path))

        return model_path

    async def __infer_local(self, image_url, model_sha1):
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
        model = self.load_model_local(model_sha1)

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

    async def __train_local(self, training_data):
        """
        Train model by CSV file
        """

        self.makedirs([self.MODELS_DIR])

        model_sha1 = training_data['model']['sha1']
        random_seed = training_data['metadata']['random_seed']

        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        self.__set_model_status(model_sha1, self.IN_PROGRESS)

        logging.warning("download training data...")
        train_dir, validate_dir, train_size, validate_size, error = await self.download_train_data(training_data)
        if error:
            logging.error("failed to download training data: {e}".format(e=error))
            self.__set_model_status(model_sha1, self.ERROR, error=error)
            return None

        logging.debug("build tf from directory. train_dir={t}, validate_dir={v}, train_size={ts}, validate_size={vs}"
                      .format(t=train_dir, v=validate_dir, ts=train_size, vs=validate_size))

        train_generator = self.__get_image_data_generator().flow_from_directory(
            train_dir,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=random_seed,
            shuffle=True,
            class_mode='categorical')
        logging.debug("train_generator tf built")

        validation_generator = self.__get_image_data_generator().flow_from_directory(
            validate_dir,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            seed=random_seed,
            shuffle=True,
            class_mode='categorical')
        logging.debug("validation_generator tf built")

        classes_count = len(train_generator.class_indices.keys())
        logging.debug("classes_count={n}".format(n=classes_count))

        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                                 input_shape=train_generator.image_shape,
                                                 kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                     seed=random_seed)))
        # model_.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model_.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu',
                                                 kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                     seed=random_seed)))
        # model_.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model_.add(tf.keras.layers.Flatten())
        model_.add(tf.keras.layers.Dense(units=64, activation='relu',
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                                             seed=random_seed)))
        model_.add(tf.keras.layers.Dense(units=classes_count, activation='softmax',
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                                             seed=random_seed)))
        logging.debug("model_ built")

        model_.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy',
                       metrics=['accuracy'])
        logging.debug("model_ compiled")

        steps_per_epoch = round(train_size) // settings.BATCH_SIZE
        validation_steps = round(validate_size) // settings.BATCH_SIZE

        model_.fit_generator(train_generator,
                             epochs=settings.EPOCHS,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             shuffle=False,
                             # max_queue_size=1,
                             callbacks=self.__get_callbacks())
        logging.debug("model_ fit")

        self.models[model_sha1]['status'] = self.READY

        model_path = self.__save_model_local(model_, train_generator.class_indices, training_data)

        # Clear TF session after train
        tf.keras.backend.clear_session()

        return model_path

    def get_model(self, model_sha1):
        """
        Tensorflow model getter from memory
        """
        empty_result = {
            'model': None,
            'class_indices': None,
            'error': None,
            'status': self.NOT_FOUND
        }
        return self.models.get(model_sha1, empty_result)

    def load_model_local(self, model_sha1):
        """
        Load model from local filesystem
        """

        model = self.get_model(model_sha1)

        if model and model.get('model') and model.get('class_indices') and model.get('status') == self.READY:
            logging.error('Successfully loading model {model_sha1}: model'.format(model_sha1=model_sha1))
            return model

        model_path = self.get_model_path(model_sha1)
        model_path_weights = os.path.join(model_path, 'model')
        model_path_json = os.path.join(model_path, 'model.json')
        model_class_indices = os.path.join(model_path, 'class_indices.json')

        if not os.path.exists(model_path) or not os.path.exists(model_path_json) or not os.path.exists(
                model_class_indices):
            logging.debug('Not loading model {model_sha1}: not all paths exists'.format(model_sha1=model_sha1))
            return model

        model = self.models[model_sha1] = {}

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

    async def load_model(self, model_uri):
        model_sha1 = get_sha1_hash(model_uri)

        try:
            self.__set_model_status(model_sha1, self.IN_PROGRESS)
            tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + '/')

            await storage_factory.read_data_from_dir(path=model_uri, path_to=tmp_path)

            model_path = self.get_model_path(model_sha1)

            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            os.makedirs(model_path, exist_ok=True)
            os.rename(tmp_path, model_path)

            self.__set_model_status(model_sha1, self.LOADING_END)

            self.load_model_local(model_sha1)
        except Exception as exc:
            self.__set_model_status(model_sha1, self.ERROR)
            logging.error('Can not download model from {model_uri}'.format(model_uri=model_uri))
            logging.error(exc)
            raise exc

    async def train(self, training_data):
        """
        Train method wrapper for support external storages for model,
        like GCS and IPFS

        TODO: add IPFS support
        """

        model = self.load_model_local(training_data['model']['sha1'])

        if model['status'] == self.READY:
            return model

        model_path = await self.__train_local(training_data)
        self.__set_model_status(training_data['model']['sha1'], self.IN_PROGRESS)
        try:
            await storage_factory.write_data_from_dir(path_from=model_path, path_to=training_data['model']['uri'])
        except Exception as exc:
            error = "Error write data to {model_uri}".format(model_uri=training_data['model']['uri'])
            self.__set_model_status(training_data['model']['sha1'], self.ERROR, error=error)
            logging.error('Cant upload data from {model_path}: {error}'.format(model_path=model_path, error=error))
            logging.error(exc)
            raise exc

        return model_path

    async def infer(self, image, model):
        """
        TODO: add IPFS support
        """

        model_sha1 = model['sha1']
        model_status = self.load_model_local(model_sha1)['status']

        if model_status == self.NOT_FOUND:
            await bgt.run(self.load_model, [model['uri']])
            model_status = self.LOADING_START

        if model_status == self.READY or model_status == self.LOADING_END:
            model_path = await self.__infer_local(image['url'], model_sha1)
            return model_path
        else:
            error = ModelIsLoading()
            error.status = model_status
            raise error
