import os

DATA_DIR = os.environ.get('DATA_DIR', 'data')
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', 120))
DATA_LIMIT = int(os.environ.get('DATA_LIMIT', 0))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 150))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
EPOCHS = int(os.environ.get('EPOCHS', 50))
TRAIN_PERCENTAGE = float(os.environ.get('TRAIN_PERCENTAGE', 0.9))
DOWNLOAD_POOL_SIZE = int(os.environ.get('DOWNLOAD_POOL_SIZE', 100))
IPFS_HOST = os.environ.get('DATA_DIR', 'https://ipfs.infura.io')
IPFS_PORT = int(os.environ.get('IPFS_PORT', 5001))
DEFAULT_INPUT_CSV_URL = os.environ.get('DEFAULT_INPUT_CSV_URL',
                                       'http://tf-models.arilot.org/static-tf-models/input.csv')
DEFAULT_MODEL_URI = os.environ.get('DEFAULT_MODEL_URI', 'default.hdf5')
DEFAULT_TEST_IMG_URL = os.environ.get('DEFAULT_TEST_IMG_URL', 'http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000014.jpg')