# static-transfer-learning

This project contains a Dockerized web service that: 

1. constructs a transfer-learning model based on a set of (image,label) pairs, and 
2. labels images based on the constructed model

The model is produced deterministically. Cryptographic hashes are produced for all inputs and outputs to ensure reproducibility and auditability.

## Motivation

When building a machine-learning system, researchers often allocate more attention to:

1. the novelty of methods employed, and
2. the performance of the system on a benchmark, than they do to (3) reproducibility of results

While this is the fastest way to publish results, the trade-off is that the productivity of the research community as a whole is reduced, as peers are unable to reproduce and assess one another's findings. 

Even more troubling is that systems built without reproducibility in mind are sometimes deployed to production environments. Business decisions are made based on ML system outputs, but results of decisions are inconsistent across trials because the ML systems are non-deterministic.

Non-determinism enters ML systems in at least two ways:

1. system initialization conditions aren't documented, and 
2. systems incorporate sources of randomness as part of their initialization process

This project demonstrates the utility of an ML system in which randomness is excluded from the build process.

## Quick Start

    docker-compose up -d
    
Please open Swagger by http://localhost:8080/api/doc

## Rest API

### Train mode

    POST /train
    
Example:
```sh
$ cat in.json 
{"csv_url": "https://storage.googleapis.com/some-bucket/some.csv"}
$ Y=`cat in.json`; curl -X POST --header 'Content-Type: application/json' --header 'Accept: text/plain' -d "$Y" http://localhost:8080/train
{"model_name": "18e5194e577513e7e60db6af9e07c58a6bbef4c8", "status": "new"}

# you can continue to issue the same command while training happens. you'll get an "in_progress" response.
$ !!
{"model_name": "18e5194e577513e7e60db6af9e07c58a6bbef4c8", "status": "in_progress"}

# eventually it finishes training.
$ !!
{"model_name": "18e5194e577513e7e60db6af9e07c58a6bbef4c8", "status": "ready"}
```

#### Arguments
**csv_url** - URL of CSV file in format:

```csv
<image_url>,<label>
```

Example:
```csv
http://tf-models.arilot.org/static-tf-models/img/Abstract-Patterned_Blouse/img_00000049.jpg,blouse
http://tf-models.arilot.org/static-tf-models/img/Abstract-Stripe_Fuzzy_Sweater/img_00000011.jpg,sweater
```

**model_url** - URL for save model file into Persistence Storage, like GCS or IPFS (no supported)

Examples:
```
gs://my-bucket/my-model                                         # Google cloud storage (not supported)
ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG/my-model  # IPFS (not supported)
``` 

#### Response

Rest API will response JSON, like

```json
{
  "model_name": "fe2199d0b79a2fe27c83c726e7b4307e1a066c02",
  "status": "ready",
}
```

**model_name** - SHA1 hash, based on CSV file URL

**status** - model status.

##### Available statuses

* **new** - Model does not exist locally and need to build
* **in_progress** - Model building process in progress
* **ready** - Model is built and ready to use

### Inference mode

    POST /inference

#### Arguments
**image_url** - URL of Image

**model_url** - Model name (from train mode response) or model URL (like GCS or IPFS) (no supported)

Examples:

```
fe2199d0b79a2fe27c83c726e7b4307e1a066c02                        # Local model
local://fe2199d0b79a2fe27c83c726e7b4307e1a066c02                # Local model example #2
gs://my-bucket/my-model                                         # Google cloud storage (not supported)
ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG/my-model  # IPFS (not supported)
```

#### Response

Predict result per label.

Example:

```json
{
  "blouse": 0.66,
  "halter": 0.0,
  "sweater": 0.34,
}
```

## Manual install

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
| EPOCHS       |  `30`         | Hyperparameter of gradient descent that controls the number of complete passes through the training dataset |
| TRAIN_PERCENTAGE | `0.9`     | Proportion of data distribution for training and validation. `0.9` means that 90% will be given for training, and 10% for training |
| DOWNLOAD_POOL_SIZE | `100`   | Size of AioPool: how many concurrent tasks can work when loading images from CSV |
| IPFS_HOST     | `https://ipfs.infura.io` | Address of IPFS endpoint. Infura public endpoint by default |
| IPFS_PORT    |  `5001`       | Port of IPFS endpoint      |
| DEFAULT_INPUT_CSV_URL | `http://tf-models.arilot.org/static-tf-models/input.csv` | Default URL of CSV file with images and labels for training. You can set this value using `--csv-url` CLI flag |
| DEFAULT_MODEL_FILENAME | `default` | Default file name of model. You can set this value using `--model-filename` CLI flag |
| DEFAULT_TEST_IMG_URL | `http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000014.jpg` | Default URL of test image for predict mode. You can set this value using `--image-url` CLI flag |
| TENSORBOARD_LOGS_ENABLED | `false` | Enable or disable logging for using ing tensorboard |
| API_HOST | `0.0.0.0` | Rest API host binding |
| API_PORT | `8080` | Rest API port binding |
| RANDOM_SEED | `1234` | Random seed for all Tensorflow methods |
