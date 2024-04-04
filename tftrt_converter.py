import importlib
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.data import AUTOTUNE


IMAGE_SIZE=(224, 224)


def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        image = preprocess(image)
        return image, label


    (training_set), ds_info = tfds.load(
        name=dataset,
        split='train',
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image).batch(batch_size)

    return training_set


def calibrated_input():
    module_name = 'densenet'
    module = prepare_module(module_name)
    preprocess_input = getattr(module, 'preprocess_input')

    training_set = prepare_dataset('cifar100', preprocess_input, 1)

    for image, label in training_set.take(128):
        yield [image]


SAVED_MODEL_DIR="./generated/cifar100/DenseNet169/saved-1"
converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=SAVED_MODEL_DIR,
        precision_mode=trt.TrtPrecisionMode.INT8,
        use_calibration=True,
)

converter.convert(calibrated_input)

converter.build(calibrated_input)

converter.save('tftrt_saved_int')
