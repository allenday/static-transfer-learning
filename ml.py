from __future__ import absolute_import, division, print_function, unicode_literals

import os
import uuid
import random
import logging
import datetime
import settings
import numpy as np
import tensorflow as tf
from datamanager import DataManager

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
os.environ['THEANO_FLAGS'] = 'dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic'


class ModelNotFound(BaseException):
    pass


class InvalidTestData(BaseException):
    pass


class ML(DataManager):
    IMG_SHAPE = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)
    model = None

    def save_model(self, file_name):
        """
        Save model to local file
        """
        # https://www.tensorflow.org/tutorials/keras/save_and_restore_models
        model_path = self.get_model_path(file_name)
        self.model.save(model_path)
        logging.info('Model saved into {model_path}'.format(model_path=model_path))
        return model_path

    def load_model(self, file_name=settings.DEFAULT_MODEL_FILENAME):
        """
        Load model from local file
        """
        model_path = self.get_model_path(file_name)
        if not os.path.exists(model_path):
            raise ModelNotFound('Model with path {model_path} not found'.format(model_path=model_path))

        return tf.keras.models.load_model(model_path)

    async def train(self, csv_url, model_uri):
        """
        Train model by CSF file
        """
        self.makedirs([self.LOG_DIR, self.MODELS_DIR])
        self.cleanup([self.TRAIN_DIR, self.VALIDATE_DIR])
        train_size, validate_size = await self.download_train_data(csv_url)

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                        zoom_range=0.2,
                                                                        horizontal_flip=True)
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                             zoom_range=0.2,
                                                                             horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        classes_count = len(train_generator.class_indices.keys())

        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Convolution2D(filters=56, kernel_size=(3, 3), activation='relu',
                                          input_shape=self.IMG_SHAPE))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=classes_count, activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])

        self.model.summary()

        # Define the Keras TensorBoard callback.
        logdir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        steps_per_epoch = round(train_size) // settings.BATCH_SIZE
        validation_steps = round(validate_size) // settings.BATCH_SIZE

        self.model.fit_generator(train_generator,
                                 epochs=settings.EPOCHS,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps,
                                 callbacks=[tensorboard_callback])

        self.model.summary()

        logging.info('Classes: {classes}'.format(classes="; ".join(
            ['%s:%s' % (i, train_generator.class_indices[i]) for i in train_generator.class_indices.keys()])))

        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator, use_multiprocessing=True)
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        return self.save_model(model_uri)

    async def inference(self, image_url, model_uri, output_uri):
        self.makedirs([self.TMP_DIR])

        self.model = self.load_model(model_uri)

        image_tmp_path = os.path.join(self.TMP_DIR, uuid.uuid1().__str__() + ".jpg")

        await self.download_file({
            "url": image_url,
            "file_path": image_tmp_path
        })

        img = tf.keras.preprocessing.image.load_img(image_tmp_path,
                                                    target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.reshape(img, [settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3])
        img = np.expand_dims(img, axis=0)

        result = self.model.predict_classes(img)

        print(result)

        os.remove(image_tmp_path)

        return result

    async def evaluate(self):
        self.model = self.load_model()

        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                             zoom_range=0.2,
                                                                             horizontal_flip=True)

        validation_generator = validation_datagen.flow_from_directory(
            self.VALIDATE_DIR,
            target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            batch_size=settings.BATCH_SIZE,
            class_mode='categorical')

        loss, categorical_accuracy, acc = self.model.evaluate(validation_generator, use_multiprocessing=True)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
