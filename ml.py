from __future__ import absolute_import, division, print_function, unicode_literals

import asyncio
import csv
import os
import logging
import uuid
import ipfsapi
import async_timeout
import tensorflow as tf
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

IMG_SHAPE = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)


class FileDownloadError(BaseException):
    pass


class ModelNotFound(BaseException):
    pass


class ML(object):
    def __init__(self):
        self.model = None
        self.ipfs_client = ipfsapi.connect(settings.IPFS_HOST, settings.IPFS_PORT)

        os.makedirs(TMP_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Setting for deterministic result
        tf.set_random_seed(1)
        tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1,
                                         use_per_session_threads=1, device_count={'CPU': 1}))

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
        return True

    def load_model(self, model_uri):
        model_path = self.__get_model_path(model_uri)
        if not os.path.exists(model_path):
            raise ModelNotFound('Model with path {model_path} not found'.format(model_path=model_path))

        return keras.models.load_model(model_path)

    async def fetch(self, session, url):
        with async_timeout.timeout(settings.HTTP_TIMEOUT):
            async with session.get(url) as response:
                return await response.read()

    async def download_file(self, session, url, file_path):
        response = await self.fetch(session, url)
        async with AIOFile(file_path, 'wb+') as afp:
            await afp.write(response)
            await afp.fsync()
            logging.info("File {url} downloaded to {file_path}".format(url=url, file_path=file_path))
            return ""

    async def get_links_for_train(self, session, csv_url, train_percentage=0.9):
        result = []

        raw_csv = await self.fetch(session, csv_url)
        csv_lines = sorted([i.decode('utf8') for i in raw_csv.splitlines()])

        total_size = len(csv_lines)
        train_size = round(total_size * train_percentage)
        validate_size = total_size - train_size

        logging.info(
            'Found {total_size} lines in csv. Train size: {train_size} / Validate size: {validate_size}'.format(
                total_size=total_size,
                train_size=train_size,
                validate_size=validate_size
            )
        )

        reader = csv.reader(csv_lines, delimiter=',', quotechar='|')
        for counter, (url, label) in enumerate(reader):
            if counter <= train_size:
                i_type = 'train'
            else:
                i_type = 'validate'

            result.append({
                'url': url,
                'label': label,
                'i_type': i_type,
                'file_name': '{counter}.jpg'.format(counter=counter)
            })

        return result

    async def download_train_data(self, csv_url):
        tasks = []

        async with ClientSession() as session:
            links = await self.get_links_for_train(session, csv_url)
            for link in links:
                dir_path = os.path.join(DATA_DIR, link['i_type'], link['label'])
                file_path = os.path.join(dir_path, link['file_name'])

                os.makedirs(dir_path, exist_ok=True)

                if not os.path.isfile(file_path):
                    tasks.append(asyncio.ensure_future(self.download_file(session, link['url'], file_path)))

            _ = await asyncio.gather(*tasks)

            logging.info('Data downloaded ({count} files)'.format(count=len(tasks)))

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        await self.download_train_data(csv_url)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='binary')

        validation_generator = validation_datagen.flow_from_directory(
            VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='binary')

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        steps_per_epoch = train_generator.n // settings.BATCH_SIZE
        validation_steps = validation_generator.n // settings.BATCH_SIZE

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=settings.EPOCHS,
                                 workers=settings.WORKERS,
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps)

        self.model.get_weights()

        self.model.summary()

        await self.save_model(model_uri)

        return True

    async def inference(self, image_url, model_uri, output_uri):
        self.model = self.load_model(model_uri)

        image_tmp_path = os.path.join(TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        async with ClientSession() as session:
            await self.download_file(session, image_url, image_tmp_path)

            img = keras.preprocessing.image.load_img(image_tmp_path,
                                                     target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            img = keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            result = self.model.predict_classes(img)

            os.remove(image_tmp_path)

            return result
