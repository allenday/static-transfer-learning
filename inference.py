#!/usr/bin/env python
import click
import logging

import settings


@click.command()
@click.option('--image-url',
              default=settings.DEFAULT_TEST_IMG_URL,
              help='Image URL')
@click.option('--model-uri', default=settings.DEFAULT_MODEL_URI, help='URI of model')
def inference(image_url, model_uri):
    import asyncio
    from ml import ML
    click.echo('Starting inference mode with URL {image_url} and model_uri {model_uri}'.format(image_url=image_url,
                                                                                               model_uri=model_uri))

    loop = asyncio.get_event_loop()

    ml = ML()

    total_future = asyncio.ensure_future(ml.inference(image_url, model_uri, None))
    loop.run_until_complete(total_future)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    inference()
