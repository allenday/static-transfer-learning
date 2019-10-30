import os
from abc import ABC

import settings
from storage.abstract import AbstractStorage


def get_real_path(path):
    if path.startswith("local://"):
        return os.path.join(settings.DATA_DIR, "models/{model}".format(model=path[len('local://'):]))

    return path


class LocalStorage(AbstractStorage, ABC):

    def read_data(self, path, path_to=None):
        """
        TODO Implement Copy to location
        :param path:
        :param dir_to:
        :return:
        """
        if path_to is not None:
            raise NotImplementedError('Copy to location is not implemented')

        with open(get_real_path(path=path), 'rb') as f:
            return f.read()

    def write_multiple_files(self, path, data):
        for element in data:
            self.write_data(os.path.join(path, element['path']), element['data'])

    def write_data(self, path, data):
        with open(get_real_path(path=get_real_path(path)), "wb") as f:
            return f.write(data)

    def read_data_from_dir(self, path, path_to=None):
        """
        TODO Implement Copy to directory
        :param path:
        :param dir_to:
        :return:
        """
        if path_to is not None:
            raise NotImplementedError('Copy to directory is not implemented')
        files = list()
        for (dirpath, dirnames, filenames) in os.walk(get_real_path(path)):
            files.extend(filenames)
            break

        data = list()
        real_path = get_real_path(path)
        for file in files:
            data.append({
                'path': file,
                'data': self.read_data(os.path.join(real_path, file))
            })

        return data

