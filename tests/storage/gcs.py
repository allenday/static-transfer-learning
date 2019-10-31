import unittest

import settings
from storage import storage_factory
from tests.storage import BaseStorage


class GcsMethods(BaseStorage, unittest.TestCase):

    """
    TODO: Crate tests
    """
    def test_write(self):
        storage_factory.write_data_from_dir(path_from="local://testmodel", path_to=settings.TEST_GOOGLE_STORAGE_MODEL_NAME)



