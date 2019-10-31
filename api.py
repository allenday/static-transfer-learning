import json
import logging
import os

import aiohttp
from aiohttp import web
from aiohttp.client_exceptions import InvalidURL
from aiohttp_swagger import setup_swagger
from aiohttp_validate import validate

import settings
from bgtask import bgt
from helpers import get_sha1_hash, get_sha1_hash_from_dir
from ml import ML, ModelNotFound, ErrorDownloadImage, ErrorProcessingImage, ModelIsLoading

m = ML()


@validate(
    request_schema={
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {"random_seed": {"type": "integer"}},
                "required": ["random_seed"],
                "additionalProperties": False
            },
            "csv": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "sha1": {"type": "string"}},
                "required": ["url", "sha1"],
                "additionalProperties": False
            },
            "model": {
                "type": "object",
                "properties": {"uri": {"type": "string"}},
                "required": ["uri"],
                "additionalProperties": False
            }
        },
        "required": ["metadata", "csv", "model"],
        "additionalProperties": False
    },
    response_schema={
        "type": "object",
        "properties": {
            "model_sha1": {"type": "string"},
            "status": {"type": "string"},
        },
    }
)
async def train(request, *args):
    """
    ---
    description: Train model
    produces:
    - application/json
    parameters:
    - in: body
      name: body
      description: Train model
      required: true
      schema:
        type: object
        properties:
          metadata:
            type: object
            properties:
              random_seed:
                type: integer
          csv:
            type: object
            properties:
              url:
                type: string
              sha1:
                type: string
          model:
            type: object
            properties:
              uri:
                type: string
    responses:
        "200":
            description: Model name and storage URL
        "405":
            description: invalid HTTP Method
        "400":
            description: Bad request
    """

    training_data = {
        'metadata': request['metadata'],
        'csv': request['csv'],
        'model': request['model']
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(training_data['csv']['url']) as resp:
            resource = await resp.text()
            training_data['csv']['content'] = resource.encode('utf-8')

    csv_content_hash = get_sha1_hash(training_data['csv']['content'])
    if training_data['csv']['sha1'] != csv_content_hash:
        logging.error("expected sha1={e}, but got sha1={g} for resource={r}".format(e=training_data['csv']['sha1'],
                                                                                    g=csv_content_hash,
                                                                                    r=training_data['csv']['url']))
        return web.Response(body=json.dumps({
            "error": "Hash of CSV does not equal of CSV content"
        }, sort_keys=True), status=400)

    model_sha1 = get_sha1_hash(str(training_data['metadata']['random_seed']).encode('utf-8'),
                               training_data['csv']['content'])
    training_data['model']['sha1'] = model_sha1
    logging.warning('model sha1={s}'.format(s=model_sha1))

    model = m.load_model_local(model_sha1)

    result = {
        'model_sha1': model_sha1,
        'status': model['status']
    }

    if model['status'] == m.NOT_FOUND:
        result['status'] = m.NEW
        await bgt.run(m.train, [training_data])
    elif model['status'] == m.ERROR and model.get('error'):
        result['error'] = model['error']
    elif model['status'] == m.READY:
        result['files_sha1'] = get_sha1_hash_from_dir(
            os.path.join(settings.DATA_DIR, 'models', training_data['model']['sha1']))

    return web.Response(body=json.dumps(result, sort_keys=True))


@validate(
    request_schema={
        "type": "object",
        "properties": {
            "image": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False
            },
            "model": {
                "type": "object",
                "properties": {"uri": {"type": "string"}, "sha1": {"type": "string"}},
                "required": ["uri", "sha1"],
                "additionalProperties": False
            }
        },
        "required": ["image", "model"],
        "additionalProperties": False
    },
    response_schema={
        "type": "object"
    }
)
async def infer(request, *args):
    """
    ---
    description: Get classes by Image URL
    produces:
    - application/json
    parameters:
    - in: body
      name: body
      description: Get classes by Image URL
      required: true
      schema:
        type: object
        properties:
          image:
            type: object
            properties:
              url:
                type: string
          model:
            type: object
            properties:
              uri:
                type: string
              sha1:
                type: string
    responses:
        "200":
            description: Labels
        "405":
            description: invalid HTTP Method
        "400":
            description: Bad request
        "500":
            description: Internal error
    """

    try:
        result = await m.infer(**request)
    except ModelNotFound:
        return web.Response(body=json.dumps({
            "error": "Model not found"
        }, sort_keys=True), status=404)
    except ModelIsLoading as e:
        return web.Response(body=json.dumps({
            "loading_status": e.status
        }), status=200)
    except InvalidURL:
        return web.Response(body=json.dumps({
            "error": "Incorrect image URL"
        }, sort_keys=True), status=400)
    except ErrorDownloadImage:
        return web.Response(body=json.dumps({
            "error": "Error download image. Please check image URL."
        }, sort_keys=True), status=500)
    except ErrorProcessingImage:
        return web.Response(body=json.dumps({
            "error": "Error processing image. Please check image URL."
        }, sort_keys=True), status=500)

    return web.Response(body=json.dumps(result))


logging.basicConfig(level=logging.INFO)

app = web.Application()
app.router.add_route('POST', "/train", train)
app.router.add_route('POST', "/infer", infer)

setup_swagger(app)

web.run_app(app, host=settings.API_HOST, port=settings.API_PORT)
