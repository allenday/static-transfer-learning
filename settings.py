import os
from distutils.util import strtobool


DATA_DIR = os.environ.get('DATA_DIR',  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', 120))
DATA_LIMIT = int(os.environ.get('DATA_LIMIT', 0))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 150))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
EPOCHS = int(os.environ.get('EPOCHS', 30))
TRAIN_PERCENTAGE = float(os.environ.get('TRAIN_PERCENTAGE', 0.9))
DOWNLOAD_POOL_SIZE = int(os.environ.get('DOWNLOAD_POOL_SIZE', 100))
IPFS_HOST = os.environ.get('DATA_DIR', 'https://ipfs.infura.io')
IPFS_PORT = int(os.environ.get('IPFS_PORT', 5001))
DEFAULT_INPUT_CSV_URL = os.environ.get('DEFAULT_INPUT_CSV_URL',
                                       'https://raw.githubusercontent.com/allenday/static-transfer-learning/master/example-data/train.csv')
DEFAULT_MODEL_FILENAME = os.environ.get('DEFAULT_MODEL_FILENAME', 'default')
DEFAULT_TEST_IMG_URL = os.environ.get('DEFAULT_TEST_IMG_URL',
                                      'https://raw.githubusercontent.com/allenday/static-transfer-learning/master/example-data/sweater/1042.jpg')
TENSORBOARD_LOGS_ENABLED = strtobool(os.environ.get('TENSORBOARD_LOGS', 'false'))
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', '8080'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '1234'))

GOOGLE_STORAGE_CREDENTIALS_PATH = os.environ.get('GOOGLE_STORAGE_CREDENTIALS_PATH',
                                      os.path.join(DATA_DIR, 'google-credentials.json'))



#TEST

TEST_GOOGLE_STORAGE_MODEL_NAME = os.environ.get('TEST_GOOGLE_STORAGE_MODEL_NAME', "gs://my-bucket/my-model")