import json
import logging
import os

import settings
from storage.abstract import AbstractStorage
from google.cloud import storage
from google.oauth2 import service_account


class GcsStorage(AbstractStorage):

    def __init__(self, credentials=None):
        if credentials:
            credentials_info = credentials
        else:
            with open(settings.GOOGLE_STORAGE_CREDENTIALS_PATH) as f:
                json_str = f.read().replace("\n", "")
                credentials_info = json.loads(json_str)

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )

        kwargs = {"project": credentials_info.get("project_id"), "credentials": credentials}
        self.client = storage.Client(**kwargs)

    def __get_bucket_name(self, path):
        return path.split("/")[2]

    def __get_file_path(self, path):
        bucket = self.__get_bucket_name(path)
        return path[len("gc://{bucket}/".format(bucket=bucket)):]

    def get_bucket(self, path):
        bucket_name = self.__get_bucket_name(path)
        return self.client.get_bucket(bucket_name)

    def write_multiple_files(self, path, data):
        for element in data:
            self.write_data(path=os.path.join(path, element['path']), data=element['data'])

    def write_data(self, path, data):
        logging.info('Start writing data to {path}'.format(path=path))
        real_path = self.__get_file_path(path)
        blob = self.get_bucket(path).blob(real_path)
        blob.upload_from_string(data)

        return blob.public_url

    def read_data(self, path):
        bucket = self.get_bucket()
        blob = bucket.blob(path)

        data = blob.download_as_string()
        return data

    def read_data_from_dir(self, path):
        """
        TODO CRETAE read_data_from_dir in google storage
        :param path:
        :return:
        """
        raise NotImplementedError()
