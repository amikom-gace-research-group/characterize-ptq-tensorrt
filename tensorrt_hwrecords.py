import pathlib
import importlib
import argparse
import time
from datetime import datetime

import silence_tensorflow.auto

import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

import tensorflow as tf
import tensorflow_datasets as tfds

# Global config
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)


def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocessor, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype=tf.float32)
        image = preprocessor(image)

        return image, label


    (test_set), ds_info = tfds.load(
        dataset,
        split='test',
        with_info=True,
        as_supervised=True)

    test_set = test_set.map(map_func=resize_image, num_parallel_calls=AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    return test_set


def do_inference(path: str, test_set):
    load_engine = EngineFromBytes(BytesFromPath(path))

    true_labels = []
    predictions = []

    with TrtRunner(load_engine) as runner:
        start_time = time.time()
        for image, label in test_set:
            true_labels.extend(label.numpy())

            outputs = runner.infer(feed_dict={'input_1': image.numpy()})
            inferred = np.argmax(outputs['dense'])

            predictions.append(inferred)

        end_time = time.time()

        elapsed = round(end_time - start_time, 2)

    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
    print(f"{round(accuracy * 100, 2)}")


def main(args):
    module = prepare_module(args.module)

    preprocess_input = getattr(module, 'preprocess_input')

    test_set = prepare_dataset(
        dataset=args.cfg,
        preprocessor=preprocess_input,
        batch_size=1
    )
    
    do_inference(args.engine, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module')
    parser.add_argument('--engine')
    parser.add_argument('--cfg')

    args = parser.parse_args()

    main(args)
