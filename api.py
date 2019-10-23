import json
import logging
import settings
from aiohttp import web
from bgtask import BackgroundTask
from aiohttp_validate import validate
from aiohttp_swagger import setup_swagger
from aiohttp.client_exceptions import InvalidURL
from ml import ML, ModelNotFound, ErrorDownloadImage, ErrorProcessingImage

m = ML()
bgt = BackgroundTask()


@validate(
    request_schema={
        "type": "object",
        "properties": {
            "csv_url": {"type": "string"},
            "model_url": {"type": "string"},
        },
        "required": ["csv_url", "model_url"],
        "additionalProperties": False
    },
    response_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "status": {"type": "string"},
        },
    }
)
async def train(request):
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
          csv_url:
            type: "string"
          model_uri:
            type: string
    responses:
        "200":
            description: Model name and storage URL
        "405":
            description: invalid HTTP Method
        "400":
            description: Bad request
    """

    data = await request.json()

    for key in ['csv_url']:
        if not data.get(key):
            return web.Response(body='Key {key} is required'.format(key=key), status=400)

    model_name = m.get_model_name(data['csv_url'])

    status = m.get_model_status(model_name)

    if status == m.NOT_FOUND:
        await bgt.run(m.train, [data['csv_url'], data.get('model_uri')])
        status = m.NEW

    return web.Response(body=json.dumps({
        'model_name': model_name,
        'status': status
    }))


@validate(
    request_schema={
        "type": "object",
        "properties": {
            "image_url": {"type": "string"},
            "model_uri": {"type": "string"},
        },
        "required": ["image_url", "model_url"],
        "additionalProperties": False
    },
    response_schema={
        "type": "object"
    }
)
async def infer(request):
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

    data = await request.json()

    for key in ['image_url', 'model_uri']:
        if not data.get(key):
            return web.Response(body='Key {key} is required'.format(key=key), status=400)

    try:
        result = await m.infer(**data)
    except ModelNotFound:
        return web.Response(body=json.dumps({
            "error": "Model not found"
        }), status=404)
    except InvalidURL:
        return web.Response(body=json.dumps({
            "error": "Incorrect image URL"
        }), status=400)
    except ErrorDownloadImage:
        return web.Response(body=json.dumps({
            "error": "Error download image. Please check image URL."
        }), status=500)
    except ErrorProcessingImage:
        return web.Response(body=json.dumps({
            "error": "Error processing image. Please check image URL."
        }), status=500)

    return web.Response(body=json.dumps(result))


logging.basicConfig(level=logging.INFO)

app = web.Application()
app.router.add_route('POST', "/train", train)
app.router.add_route('POST', "/infer", infer)

setup_swagger(app)

web.run_app(app, host=settings.API_HOST, port=settings.API_PORT)
