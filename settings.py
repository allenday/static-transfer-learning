import os

DATA_DIR = os.environ.get('DATA_DIR', 'data')
HTTP_TIMEOUT = int(os.environ.get('HTTP_TIMEOUT', 120))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 160))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))
EPOCHS = int(os.environ.get('EPOCHS', 10))
WORKERS = int(os.environ.get('WORKERS', 1))
IPFS_HOST = os.environ.get('DATA_DIR', 'https://ipfs.infura.io')
IPFS_PORT = os.environ.get('IPFS_PORT', 5001)