from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import datetime
import os
import logging
import random
import shutil
import uuid
import ipfsapi
import async_timeout
import tensorflow as tf
from asyncio_pool import AioPool
from tensorflow import keras
from aiofile import AIOFile
from aiohttp import ClientSession
import numpy as np

import settings

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, settings.DATA_DIR)
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATE_DIR = os.path.join(DATA_DIR, 'validate')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
TMP_DIR = os.path.join(DATA_DIR, 'tmp')
LOG_DIR = os.path.join(DATA_DIR, 'logs')

IMG_SHAPE = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)


class FileDownloadError(BaseException):
    pass


class ModelNotFound(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ML(object):
    def __init__(self):
        self.model = None
        self.ipfs_client = ipfsapi.connect(settings.IPFS_HOST, settings.IPFS_PORT)
        self.pool = AioPool(size=settings.DOWNLOAD_POOL_SIZE)

        # Setting for deterministic result
        tf.compat.v1.set_random_seed(1)
        # self.tf_session = tf.Session(
        #     config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1,
        #                           use_per_session_threads=1, device_count={'CPU': 1}))

        self.__makedirs()

    def __makedirs(self):
        for path in (TMP_DIR, MODELS_DIR, TRAIN_DIR, VALIDATE_DIR, LOG_DIR):
            os.makedirs(path, exist_ok=True)

    def __ipfs_save(self, file_path):
        return self.ipfs_client.add(file_path)

    def __get_ipfs(self, hash):
        return self.ipfs_client.cat(hash)

    def download(self, hash, path):
        with open(path, 'wb+') as f:
            f.write(self.__get_ipfs(hash))
        return path

    @staticmethod
    def __get_model_path(model_uri):
        return os.path.join(MODELS_DIR, model_uri)

    async def save_model(self, model_uri):
        # https://www.tensorflow.org/tutorials/keras/save_and_restore_models
        model_path = self.__get_model_path(model_uri)
        self.model.save(model_path)
        logging.info('Model saved into {model_path}'.format(model_path=model_path))
        return model_path

    def load_model(self, model_uri=settings.DEFAULT_MODEL_URI):
        model_path = self.__get_model_path(model_uri)
        if not os.path.exists(model_path):
            raise ModelNotFound('Model with path {model_path} not found'.format(model_path=model_path))

        return keras.models.load_model(model_path)

    async def fetch(self, url):
        async with ClientSession() as session:
            with async_timeout.timeout(settings.HTTP_TIMEOUT):
                async with session.get(url) as response:
                    return await response.read()

    async def download_file(self, req):
        url = req['url']
        file_path = req['file_path']

        logging.debug('Downloading {url} into {file_path}'.format(url=url, file_path=file_path))

        response = await self.fetch(url)
        try:
            async with AIOFile(file_path, 'wb+') as afp:
                await afp.write(response)
                await afp.fsync()
                logging.info("File {url} downloaded to {file_path}".format(url=url, file_path=file_path))
        except:
            logging.warning("Error downloading file {url}".format(url=url))
            os.remove(file_path)

        if not os.stat(file_path).st_size:
            logging.warning("File {file_path} is empty".format(file_path=file_path))
            os.remove(file_path)

        return True

    async def get_links_for_train(self, csv_url):
        result = []
        result_by_labels = {}

        raw_csv = await self.fetch(csv_url)
        csv_lines = [i.decode('utf8') for i in raw_csv.splitlines()]
        random.seed(1)
        random.shuffle(csv_lines)

        reader = csv.reader(csv_lines, delimiter=',', quotechar='|')
        for url, label in reader:
            if label not in result_by_labels:
                result_by_labels[label] = []

            result_by_labels[label].append(url)

        labels_count = len(result_by_labels.keys())

        logging.info("Found {labels_count} labels in CSV".format(labels_count=labels_count))

        label_imgs_limit = min([len(result_by_labels[i]) for i in result_by_labels])

        if settings.DATA_LIMIT and label_imgs_limit > settings.DATA_LIMIT:
            label_imgs_limit = settings.DATA_LIMIT

        train_size = round(label_imgs_limit * settings.TRAIN_PERCENTAGE)
        validate_size = label_imgs_limit - train_size

        logging.info(
            'Found {label_imgs_limit} lines in csv. Train size: {train_size} / Validate size: {validate_size}'.format(
                label_imgs_limit=label_imgs_limit,
                train_size=train_size,
                validate_size=validate_size
            )
        )

        for counter in range(0, label_imgs_limit):
            if counter <= train_size:
                i_type = 'train'
            else:
                i_type = 'validate'

            for label in result_by_labels.keys():
                url = result_by_labels[label].pop()
                result.append({
                    'url': url,
                    'label': label,
                    'i_type': i_type,
                    'file_name': '{counter}.jpg'.format(counter=counter)
                })

        return result

    async def cleanup(self):
        for path in [TRAIN_DIR, VALIDATE_DIR]:
            shutil.rmtree(path, ignore_errors=True)

    async def download_train_data(self, csv_url):
        tasks = []
        created_label_dirs = []

        links = await self.get_links_for_train(csv_url)
        logging.info('Found {count} links'.format(count=len(links)))

        for link in links:
            dir_path = os.path.join(DATA_DIR, link['i_type'], link['label'])
            file_path = os.path.join(dir_path, link['file_name'])

            if dir_path not in created_label_dirs:
                os.makedirs(dir_path, exist_ok=True)
                created_label_dirs.append(dir_path)

            if not os.path.isfile(file_path):
                tasks.append({
                    "url": link['url'],
                    "file_path": file_path
                })

        if tasks:
            await self.pool.map(self.download_file, tasks)

        count = len(tasks)
        logging.info('Data downloaded ({count} files)'.format(count=count))

        return count

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        await self.cleanup()
        num_train = await self.download_train_data(csv_url)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                     zoom_range=0.2,
                                                                     horizontal_flip=True)
        validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                          zoom_range=0.2,
                                                                          horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu', input_shape=IMG_SHAPE))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=64, activation='relu'))
        self.model.add(keras.layers.Dense(units=classes_count, activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])

        self.model.summary()

        # Define the Keras TensorBoard callback.
        logdir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        steps_per_epoch = round(num_train) // settings.BATCH_SIZE
        self.model.fit_generator(train_generator,
                                 epochs=settings.EPOCHS,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_generator,
                                 validation_steps=settings.VALIDATION_STEPS,
                                 callbacks=[tensorboard_callback])

        self.model.summary()

        logging.info('Classes: {classes}'.format(classes="; ".join(
            ['%s:%s' % (i, train_generator.class_indices[i]) for i in train_generator.class_indices.keys()])))

        print(self.model.evaluate_generator(validation_generator))

        return await self.save_model(model_uri)

    async def inference(self, image_url, model_uri, output_uri):
        self.model = self.load_model(model_uri)

        image_tmp_path = os.path.join(TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        await self.download_file({
            "url": image_url,
            "file_path": image_tmp_path
        })

        img = keras.preprocessing.image.load_img(image_tmp_path,
                                                 target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.reshape(img, [settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3])
        img = np.expand_dims(img, axis=0)

        result = self.model.predict_classes(img)

        print(result)

        os.remove(image_tmp_path)

        return result
