#!/usr/bin/env python
import click
import logging
import settings


@click.command()
@click.option('--model-filename', default=settings.DEFAULT_MODEL_FILENAME, help='File name of model')
def evaluate(model_filename):
    import asyncio
    from ml import ML

    loop = asyncio.get_event_loop()

    ml = ML()

    total_future = asyncio.ensure_future(ml.evaluate(model_filename))
    loop.run_until_complete(total_future)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    evaluate()
