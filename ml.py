from __future__ import absolute_import, division, print_function, unicode_literals

import asyncio
import csv
import os
import logging
import random
import uuid
import matplotlib.pyplot as plt
import ipfsapi
import async_timeout
import tensorflow as tf
from asyncio_pool import AioPool
from tensorflow import keras
from aiofile import AIOFile
from aiohttp import ClientSession
import numpy as np
import PIL
from PIL import Image

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


class InvalidTestData(BaseException):
    pass


class ML(object):
    def __init__(self):
        self.model = None
        self.ipfs_client = ipfsapi.connect(settings.IPFS_HOST, settings.IPFS_PORT)
        self.pool = AioPool(size=settings.DOWNLOAD_POOL_SIZE)

        # Setting for deterministic result
        tf.set_random_seed(1)
        # self.tf_session = tf.Session(
        #     config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1,
        #                           use_per_session_threads=1, device_count={'CPU': 1}))

        self.__makedirs()

    def __makedirs(self):
        for path in (TMP_DIR, MODELS_DIR, TRAIN_DIR, VALIDATE_DIR):
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
        return True

    def load_model(self, model_uri):
        model_path = self.__get_model_path(model_uri)
        if not os.path.exists(model_path):
            raise ModelNotFound('Model with path {model_path} not found'.format(model_path=model_path))

        return keras.models.load_model(model_path)

    @staticmethod
    def jpeg_to_8_bit_greyscale(path):
        """
        Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
        """
        img = Image.open(path).convert('L')  # convert image to 8-bit grayscale
        # Make aspect ratio as 1:1, by applying image crop.
        # Please note, croping works for this data set, but in general one
        # needs to locate the subject and then crop or scale accordingly.
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
            m_min_d = min(WIDTH, HEIGHT)
            img = img.crop((0, 0, m_min_d, m_min_d))
        # Scale the image to the requested maxsize by Anti-alias sampling.
        img.thumbnail((settings.IMAGE_SIZE, settings.IMAGE_SIZE), PIL.Image.ANTIALIAS)
        return np.asarray(img)

    async def fetch(self, url):
        async with ClientSession() as session:
            with async_timeout.timeout(settings.HTTP_TIMEOUT):
                async with session.get(url) as response:
                    return await response.read()

    async def download_file(self, req):
        url = req['url']
        file_path = req['file_path']

        print('Downloading {url} into {file_path}'.format(url=url, file_path=file_path))

        response = await self.fetch(url)
        try:
            async with AIOFile(file_path, 'wb+') as afp:
                await afp.write(response)
                await afp.fsync()
                logging.info("File {url} downloaded to {file_path}".format(url=url, file_path=file_path))
                return ""
        except:
            os.remove(file_path)

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

        print("Found {labels_count} labels in CSV".format(labels_count=labels_count))

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

    async def download_train_data(self, csv_url):
        tasks = []
        created_label_dirs = []

        links = await self.get_links_for_train(csv_url)
        print('Found {count} links'.format(count=len(links)))

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

        await self.pool.map(self.download_file, tasks)

        logging.info('Data downloaded ({count} files)'.format(count=len(tasks)))

    async def load_image_dataset(self, path_dir):
        images = []
        labels = []

        class_names = os.listdir(path_dir)
        class_names.sort()

        for label in class_names:
            for file in os.listdir(os.path.join(path_dir, label)):
                img = self.jpeg_to_8_bit_greyscale(os.path.join(path_dir, label, file))
                images.append(img)
                labels.append(class_names.index(label))

        return np.asarray(images), np.asarray(labels), class_names

    async def display_images(self, images, labels, class_names):
        plt.figure(figsize=(10, 10))
        grid_size = min(25, len(images))
        for i in range(grid_size):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[labels[i]])
        plt.show()

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        await self.download_train_data(csv_url)

        train_images, train_labels, train_class_names = await self.load_image_dataset(TRAIN_DIR)
        test_images, test_labels, test_class_names = await self.load_image_dataset(VALIDATE_DIR)

        if train_class_names != test_class_names:
            raise InvalidTestData("Labels in training and validate data do not match")

        print(train_images.shape)
        print(train_labels)

        # await self.display_images(train_images, train_labels, train_class_names)

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Setting up the layers.
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(train_images, train_labels, epochs=settings.EPOCHS)

        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

        predictions = self.model.predict(test_images)

        print(predictions)

        await self.display_images(test_images, np.argmax(predictions, axis=1), class_names=train_class_names)

        await self.save_model(model_uri)

        return True

    async def inference(self, image_url, model_uri, output_uri):
        self.model = self.load_model(model_uri)

        image_tmp_path = os.path.join(TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        await self.download_file(image_url, image_tmp_path)

        img = keras.preprocessing.image.load_img(image_tmp_path,
                                                 target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        result = self.model.predict_classes(img)

        os.remove(image_tmp_path)

        print(result)

        return result
