from abc import ABC, abstractmethod


class AbstractStorage(ABC):

    @abstractmethod
    def write_multiple_files(self, path, data):
        """Write multiples data to path
            Args:
                path (str): Path
                data (list): List of file. Format:
                    {
                        "data": data (bytes)
                        "name": name (str)
                    }
        """
        pass

    @abstractmethod
    def write_data(self, path, data):
        pass

    @abstractmethod
    def read_data(self, path, path_to=None):
        pass

    @abstractmethod
    def read_data_from_dir(self, path, path_to=None):
        pass
