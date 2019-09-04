#!/usr/bin/env python
import click
import logging


@click.command()
def evaluate():
    import asyncio
    from ml import ML

    loop = asyncio.get_event_loop()

    ml = ML()

    total_future = asyncio.ensure_future(ml.evaluate())
    loop.run_until_complete(total_future)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    evaluate()
