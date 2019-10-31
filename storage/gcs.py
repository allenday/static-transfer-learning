import asyncio
import logging
import os

from gcloud.aio.storage import Storage, Bucket
from aiohttp import ClientSession as Session

import settings
from storage.abstract import AbstractStorage


class GcsStorage(AbstractStorage):
    def __get_bucket_name(self, path):
        return path.split("/")[2]

    def __get_file_path(self, path):
        bucket = self.__get_bucket_name(path)
        return path[len("gc://{bucket}/".format(bucket=bucket)):]

    async def write_multiple_files(self, path, data):
        futures = []
        for element in data:
            futures.append(self.write_data(path=os.path.join(path, element['path']), data=element['data']))
        await asyncio.gather(*futures)

    async def write_data(self, path, data):
        logging.debug('Start writing data to {path}'.format(path=path))
        real_path = self.__get_file_path(path)
        bucket_name = self.__get_bucket_name(path)

        async with Session(timeout=settings.GOOGLE_CLOUD_STORAGE_UPLOAD_TIMEOUT) as session:
            storage = Storage(service_file=settings.GOOGLE_APPLICATION_CREDENTIALS, session=session)
            return await storage.upload(bucket_name, real_path, data,
                                        timeout=settings.GOOGLE_CLOUD_STORAGE_UPLOAD_TIMEOUT)

    async def read_data(self, path, path_to=None):
        bucket_name = self.__get_bucket_name(path)
        async with Session(timeout=settings.GOOGLE_CLOUD_STORAGE_UPLOAD_TIMEOUT) as session:
            storage = Storage(service_file=settings.GOOGLE_APPLICATION_CREDENTIALS, session=session)
            data = await storage.download(bucket_name, self.__get_file_path(path),
                                          timeout=settings.GOOGLE_CLOUD_STORAGE_UPLOAD_TIMEOUT)

            if path_to:
                os.makedirs(os.path.dirname(path_to), exist_ok=True)
                open(path_to, 'wb').write(data)

            return data

    async def read_data_from_dir(self, path, path_to=None):
        bucket_name = self.__get_bucket_name(path)
        async with Session() as session:
            storage = Storage(service_file=settings.GOOGLE_APPLICATION_CREDENTIALS, session=session)
            bucket = Bucket(storage, bucket_name)

            items = list(await bucket.list_blobs(prefix=self.__get_file_path(path)))

            result = list()
            base_file_path = self.__get_file_path(path)
            for item in items:
                file_location = item[len(base_file_path) + 1:]
                path_to_location = os.path.join(path_to, file_location) if path_to else None
                result.append({
                    "path": file_location,
                    "location": path_to_location,
                    "data": await self.read_data(path=os.path.join(path, file_location), path_to=path_to_location)
                })

            return result
