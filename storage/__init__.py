from abc import ABC

from storage.abstract import AbstractStorage
from storage.gcs import GcsStorage
from storage.local import LocalStorage


class Storage(AbstractStorage, ABC):
    __gsc_storage = None
    __local_storage = None

    def __get_gcs_storage(self):
        if self.__gsc_storage is None:
            self.__gsc_storage = GcsStorage()

        return self.__gsc_storage

    def __get_local_storage(self):
        if self.__local_storage is None:
            self.__local_storage = LocalStorage()

        return self.__local_storage

    def get_storage(self, path):
        if path.startswith("gs://"):
            return self.__get_gcs_storage()

        if path.startswith("local://"):
            return self.__get_local_storage()

    def write_data_from_dir(self, path_from, path_to):
        data = self.read_data_from_dir(path_from)
        return self.get_storage(path_to).write_multiple_files(path=path_to, data=data)

    def read_data_from_dir(self, path):
        return self.get_storage(path).read_data_from_dir(path)


    def write_multiple_files(self, path, data):
        path

    def write_data(self, path, data):
        self.get_storage(path).write_data(path, data)

    def read_data(self, path):
        return self.get_storage(path).read_data(path)


storage_factory = Storage()
