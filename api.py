from aiohttp import web
from aiohttp.client_exceptions import InvalidURL
from aiohttp_swagger import setup_swagger
from aiohttp_validate import validate
from bgtask import BackgroundTask
from ml import ML, ModelNotFound, ErrorDownloadImage, ErrorProcessingImage
import hashlib
import json
import logging
import settings
import urllib.request

m = ML()
bgt = BackgroundTask()

@validate(
    request_schema={
        "type": "object",
        "properties": {
            "metadata": {
              "type": "object",
              "properties": { "random_seed": {"type": "integer"} },
              "required": [ "random_seed" ],
              "additionalProperties": False
            },
            "csv": {
              "type": "object",
              "properties": { "url": {"type": "string"}, "uri": {"type": "string"} },
              "required": [ "url", "uri" ],
              "additionalProperties": False
            },
            "model": {
              "type": "object",
              "properties": { "name": {"type": "string"}, "uri": {"type": "string"} },
              "required": [ "name", "uri" ],
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
              uri:
                type: string
          model:
            type: object
            properties:
              name:
                type: string
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

    training_data = { 'metadata': request['metadata'], 'csv': request['csv'], 'model': request['model'] }


    resource = urllib.request.urlopen(training_data['csv']['url'])
    training_data['csv']['content'] = resource.read().decode(resource.headers.get_content_charset()).encode('utf-8')

    m0 = hashlib.sha1()
    m0.update(training_data['csv']['content'])
    if training_data['csv']['uri'] != m0.hexdigest():
        logging.error("expected sha1={e}, but got sha1={g} for resource={r}".format(e=training_data['csv']['uri'],g=m0.hexdigest(),r=training_data['csv']['url']))
        return None

    m1 = hashlib.sha1()
    m1.update(str(training_data['metadata']['random_seed']).encode('utf-8'))
    m1.update(training_data['csv']['content'])
    training_data['model']['uri'] = m1.hexdigest()
    logging.warn('model sha1={s}'.format(s=m1.hexdigest()))

    model_sha1 = training_data['model']['uri']

    model = m.get_model(model_sha1)
    model_status = model['status']

    result = {
        'model_sha1': model_sha1,
        'status': model_status
    }

    if model_status == m.NOT_FOUND:
        result['status'] = m.NEW
        await bgt.run(m.train, [training_data])
    elif model_status == m.ERROR and model.get('error'):
        result['error'] = model['error']

    return web.Response(body=json.dumps(result,sort_keys=True))


@validate(
    request_schema={
        "type": "object",
        "properties": {
            "image_url": {"type": "string"},
            "model_uri": {"type": "string"},
        },
        "required": ["image_url", "model_uri"],
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
          image_url:
            type: "string"
          model_uri:
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
        },sort_keys=True), status=404)
    except InvalidURL:
        return web.Response(body=json.dumps({
            "error": "Incorrect image URL"
        },sort_keys=True), status=400)
    except ErrorDownloadImage:
        return web.Response(body=json.dumps({
            "error": "Error download image. Please check image URL."
        },sort_keys=True), status=500)
    except ErrorProcessingImage:
        return web.Response(body=json.dumps({
            "error": "Error processing image. Please check image URL."
        },sort_keys=True), status=500)

    return web.Response(body=json.dumps(result))


logging.basicConfig(level=logging.INFO)

app = web.Application()
app.router.add_route('POST', "/train", train)
app.router.add_route('POST', "/infer", infer)

setup_swagger(app)

web.run_app(app, host=settings.API_HOST, port=settings.API_PORT)
