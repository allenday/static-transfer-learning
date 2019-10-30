from __future__ import absolute_import, division, print_function, unicode_literals
from aiofile import AIOFile
from aiohttp import ClientSession
from asyncio_pool import AioPool
import async_timeout
import csv
import hashlib
import logging
import os
import random
import settings
import shutil
import ipfshttpclient

from helpers import get_sha1_hash


class InvalidTrainingData(BaseException):
    pass


class DataManager(object):
    MIN_TRAIN_SIZE = 90
    MIN_VALIDATE_SIZE = 10
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, settings.DATA_DIR)
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALIDATE_DIR = os.path.join(DATA_DIR, 'validate')
    TMP_DIR = os.path.join(DATA_DIR, 'tmp')
    LOG_DIR = os.path.join(DATA_DIR, 'logs')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')

    def __init__(self):
        self.ipfs_client = ipfshttpclient.connect(settings.IPFS_ADDRESS)

    def __ipfs_save(self, file_path):
        return self.ipfs_client.add(file_path)

    def __get_ipfs(self, hash):
        return self.ipfs_client.cat(hash)

    def get_model_path(self, model_sha1):
        return os.path.join(self.MODELS_DIR, model_sha1)

    def makedirs(self, dirs):
        for path in dirs:
            os.makedirs(path, exist_ok=True)

    def cleanup(self, dirs):
        for path in dirs:
            shutil.rmtree(path, ignore_errors=True)

    def download(self, hash, path):
        with open(path, 'wb+') as f:
            f.write(self.__get_ipfs(hash))
        return path

    async def fetch(self, url):
        async with ClientSession() as session:
            with async_timeout.timeout(settings.HTTP_TIMEOUT):
                async with session.get(url) as response:
                    return await response.read()

    async def download_file(self, req):
        url = req['url']
        sha1 = req.get('sha1')
        file_path = req['file_path']

        logging.debug('Downloading {url} into {file_path}'.format(url=url, file_path=file_path))

        response = await self.fetch(url)
        try:
            async with AIOFile(file_path, 'wb+') as afp:
                await afp.write(response)
                await afp.fsync()

                if sha1:
                    image_hash = get_sha1_hash(response)
                    if image_hash != sha1:
                        e = "incorrect sha1 for {f}".format(f=url)
                        logging.error(e)
                        raise Exception(e)
                    logging.debug("File {url} downloaded to {file_path}".format(url=url, file_path=file_path))
        except Exception as exc:
            logging.error("Error downloading file {url}: {exc}".format(url=url, exc=str(exc)))
            os.remove(file_path)
            return False

        if not os.stat(file_path).st_size:
            logging.error("File {file_path} is empty".format(file_path=file_path))
            os.remove(file_path)
            return False

        return True

    async def get_links_for_train(self, training_data):
        links = []
        result_by_labels = {}

        raw_csv = training_data['csv']['content']
        csv_lines = [i.decode('utf8') for i in raw_csv.splitlines()]

        # TODO allenday
        random.seed(settings.RANDOM_SEED)
        random.shuffle(csv_lines)

        reader = csv.reader(csv_lines, delimiter=',', quotechar='|')
        csv_lines_counter = 0
        for url, label, sha1 in reader:
            csv_lines_counter += 1
            if label not in result_by_labels:
                result_by_labels[label] = []

            result_by_labels[label].append([url, sha1])

        labels_count = len(result_by_labels.keys())

        logging.info("Found {labels_count} labels in CSV".format(labels_count=labels_count))

        label_imgs_limit = min([len(result_by_labels[i]) for i in result_by_labels])

        if settings.DATA_LIMIT and label_imgs_limit > settings.DATA_LIMIT:
            label_imgs_limit = settings.DATA_LIMIT

        train_size = round(label_imgs_limit * settings.TRAIN_PERCENTAGE)
        validate_size = label_imgs_limit - train_size

        logging.info(
            'Found {csv_lines_counter} lines in csv. Train size: {train_size} / Validate size: {validate_size}'.format(
                csv_lines_counter=csv_lines_counter,
                train_size=train_size * labels_count,
                validate_size=validate_size * labels_count
            )
        )
        logging.info('Train items count per label: %d' % train_size)
        logging.info('Validate items count per label: %d' % validate_size)

        for counter in range(0, label_imgs_limit):
            if counter <= train_size:
                i_type = 'train'
            else:
                i_type = 'validate'

            for label in result_by_labels.keys():
                dat = result_by_labels[label].pop()
                links.append({
                    'url': dat[0],
                    'sha1': dat[1],
                    'label': label,
                    'i_type': i_type,
                    'file_name': '{counter}.jpg'.format(counter=counter)
                })

        logging.info('Found {count} links'.format(count=len(links)))

        return links, train_size, validate_size

    async def download_train_data(self, training_data):
        tasks = []
        created_label_dirs = []

        model_sha1 = training_data['model']['sha1']

        train_dir = os.path.join(self.TRAIN_DIR, model_sha1)
        validate_dir = os.path.join(self.VALIDATE_DIR, model_sha1)

        logging.debug(
            "model_sha1={m}, train_dir={t}, validate_dir={v}".format(m=model_sha1, t=train_dir, v=validate_dir))

        self.cleanup([train_dir, validate_dir])
        self.makedirs([train_dir, validate_dir])

        try:
            logging.info("get links for train")
            links, train_size, validate_size = await self.get_links_for_train(training_data)
        except Exception as exc:
            error = "Error extract data from csv-file"
            logging.error('Cant train model by data from csv {csv_url}, random_seed {r}: {error}'.format(
                csv_url=training_data['csv']['url'], r=training_data['metadata']['random_seed'], error=error))
            logging.error(exc)
            return None, None, None, None, error

        # Custom validation (https://github.com/OlafenwaMoses/ImageAI/issues/294)
        if train_size < self.MIN_TRAIN_SIZE or validate_size < self.MIN_VALIDATE_SIZE:
            error = "You should have at least 300 for train and 100 for test per label."
            logging.error(
                'Cant train model by data from csv {csv_url}: {error}'.format(csv_url=training_data['csv']['url'],
                                                                              error=error))
            return None, None, None, None, error

        for link in links:
            dir_path = os.path.join(self.DATA_DIR, link['i_type'], model_sha1, link['label'])
            file_path = os.path.join(dir_path, link['file_name'])

            # logging.info("processing link url={u}, sha1={s}, path={p}".format(u=link['url'], s=link['sha1'], p=file_path))

            if dir_path not in created_label_dirs:
                os.makedirs(dir_path, exist_ok=True)
                created_label_dirs.append(dir_path)

            if not os.path.isfile(file_path):
                tasks.append({
                    "url": link['url'],
                    "sha1": link['sha1'],
                    "file_path": file_path
                })

        if tasks:
            pool = AioPool(size=settings.DOWNLOAD_POOL_SIZE)
            await pool.map(self.download_file, tasks)

        logging.info('Data downloaded ({count} files)'.format(count=len(tasks)))

        return train_dir, validate_dir, train_size, validate_size, None
