# static-transfer-learning

## Install

## Using Docker

    docker build -t static-transfer-learning .

## Using PIP

Python 3.7+ required.

    pip install -r requirements.txt

## Configure

The entire configuration is set through a environment variables.

Available variables:

| Variable     |     Default   |  Description               |
|--------------|:-------------:|---------------------------:|
| DATA_DIR     |  `data`       | Path to data dir           |
| HTTP_TIMEOUT |  `120`        | Timeout of http connection |
| DATA_LIMIT   |  `0`          | Max elements per label. If value is `0`, If the value is 0, then the minimum available value from all labels will be selected |
| IMAGE_SIZE   |  `150`        | Width and height of the picture for cropping |
| BATCH_SIZE   |  `32`         | Number of training samples to work through before the model’s internal parameters are updated |
| EPOCHS       |  `67`         | Hyperparameter of gradient descent that controls the number of complete passes through the training dataset |
| TRAIN_PERCENTAGE | `0.9`     | Proportion of data distribution for training and validation. `0.9` means that 90% will be given for training, and 10% for training |
| DOWNLOAD_POOL_SIZE | `100`   | Size of AioPool: how many concurrent tasks can work when loading images from CSV |
| IPFS_HOST     | `https://ipfs.infura.io` | Address of IPFS endpoint. Infura public endpoint by default |
| IPFS_PORT    |  `5001`       | Port of IPFS endpoint      |
| DEFAULT_INPUT_CSV_URL | `http://tf-models.arilot.org/static-tf-models/input.csv` | Default URL of CSV file with images and labels for training. You can set this value using `--csv-url` CLI flag |
| DEFAULT_MODEL_FILENAME | `default.hdf5` | Default file name of model. You can set this value using `--model-filename` CLI flag |
| DEFAULT_TEST_IMG_URL | `http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000014.jpg` | Default URL of test image for predict mode. You can set this value using `--image-url` CLI flag |

## Usage

### Training mode

    python train.py --csv-url=http://tf-models.arilot.org/static-tf-models/input.csv --model-uri=mymodel.hdf5

Docker way:

    docker run --rm -v /path/to/data/dir:/usr/src/app/data static-transfer-learning python train.py --csv-url=http://tf-models.arilot.org/static-tf-models/input.csv --model-uri=mymodel.hdf5

### Evaluate mode

    python evaluate.py

Docker way:

    docker run --rm -v /path/to/data/dir:/usr/src/app/data static-transfer-learning python evaluate.py

### Inference mode (in progress)

    python inference.py --model-uri=mymodel.hdf5 --image-url=http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000014.jpg

Docker way:

    docker run --rm -v /path/to/data/dir:/usr/src/app/data static-transfer-learning python inference.py --model-uri=mymodel.hdf5 --image-url=http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000014.jpg
