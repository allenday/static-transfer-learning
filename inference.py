#!/usr/bin/env python
import click


@click.command()
@click.option('--image-url',
              default='http://tf-models.arilot.org/static-tf-models/img/Embroidered_Gauze_Blouse/img_00000112.jpg',
              help='Image URL')
@click.option('--model-uri', default='default', help='URI of model')
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
    inference()
