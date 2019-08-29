#!/usr/bin/env python
import click


@click.command()
@click.option('--csv-url', default='http://127.0.0.1:5000/static/input.csv', help='CSV file url')
@click.option('--model-uri', default='default', help='URI of model')
def train(csv_url, model_uri):
    import asyncio
    from ml import ML
    click.echo('Starting train mode with URL {csv_url} and model_uri {model_uri}'.format(csv_url=csv_url,
                                                                                         model_uri=model_uri))

    loop = asyncio.get_event_loop()

    ml = ML()

    total_future = asyncio.ensure_future(ml.train(csv_url, model_uri))
    loop.run_until_complete(total_future)


if __name__ == '__main__':
    train()
