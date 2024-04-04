import threading
import importlib
# import numpy as np
import argparse
import subprocess
from datetime import datetime
from timeit import default_timer as timer
import time
import os
import psutil
import pathlib
import re
import silence_tensorflow.auto
import tensorflow as tf


# TensorFlow GPU config
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Threads
class CPU(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1'])
                output = output.decode("utf-8")
                cpu_ = float(output.splitlines()[-1].split()[-3])
                self._list.append(cpu_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


class Memory(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1', '-r'])
                output = output.decode("utf-8")
                mem_ = float(output.splitlines()[-1].split()[-3])
                self._list.append(mem_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


def _jstat_start():
    subprocess.check_output(
        f'tegrastats --interval 1000 --start --logfile test.txt',
        shell=True)


def _jstat_stop():
    subprocess.check_output(f'tegrastats --stop', shell=True)
    out = open("test.txt", 'r')
    lines = out.read().split('\n')
    entire = []
    try:
        for line in lines:
            pattern = r"GR3D_FREQ (\d+)%"
            match = re.search(pattern, line)
            if match:
                gpu_ = match.group(1)
                entire.append(float(gpu_))
        # entire = [num for num in entire if num > 10.0]
        result = sum(entire) / len(entire)
    except:
        result = 0
        entire = entire
        pass

    subprocess.check_output("rm test.txt", shell=True)
    return result, entire


def process_memory():
    process = psutil.Process(pid=os.getpid())
    return process.memory_info().rss


def build_model(model_name, backbone, num_output, weights):
    if model_name in ['MobileNetV3Small', 'MobileNetV3Large']:
        backbone = backbone(include_top=False,
                            input_shape=(224, 224, 3),
                            pooling="avg",
                            minimalistic=True)
    else:
        backbone = backbone(include_top=False,
                            input_shape=(224, 224, 3),
                            pooling="avg")

    backbone.trainable = False

    outputs = tf.keras.layers.Dense(num_output, activation="softmax")(backbone.output)
    model = tf.keras.Model(backbone.input, outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def prepare_data(batch):
    IMG_SIZE = (224, 224)

    image_dir = f"{pathlib.Path.cwd()}/assets/"
    image_dataset = tf.keras.utils.image_dataset_from_directory(image_dir,
                                                                image_size=IMG_SIZE,
                                                                batch_size=batch)

    return image_dataset


def calc_latency_per_batch(inference_time, batch):
    latency_s = (inference_time * batch) / 2500  # Latency in seconds!
    latency_ms = round(latency_s * 1000, 2)  # In miliseconds!

    return latency_s, latency_ms


def infer(model, batch, images):
    # Threading starts
    cpu_thread = CPU()
    mem_thread = Memory()
    cpu_thread.start()
    mem_thread.start()
    _jstat_start()

    begin = datetime.now()
    model.predict(images, verbose=0, batch_size=batch)
    gpu = float(_jstat_stop()[0])
    end = datetime.now()

    delta = end - begin

    cpu_thread.stop()
    mem_thread.stop()
    cpu_thread.join()
    mem_thread.join()

    cpu_use = round(cpu_thread.result[0], 2)
    mem_use = round((mem_thread.result[0] / 1024), 2)
    # memuse_2 = round(float(memuse_2) / 1024 / 1024, 2)
    gpu = round(gpu, 2)

    latency_s, latency_ms = calc_latency_per_batch(delta.total_seconds(), batch) 
    print(f"{delta.total_seconds()},{latency_ms},{cpu_use},{gpu},{mem_use}")


def main(args):
    # clear cache
    os.system("echo 'CloudLab12#$%' | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")

    module = importlib.import_module("tensorflow.keras.applications")
    backbone = getattr(module, args.model_name)
    num_output = 102 if args.cfg == "oxford_flowers102" else 100

    model = build_model(args.model_name, backbone, num_output, args.weights)
    mem_after_model_load = process_memory()
    mem_after_model_load = round(float(mem_after_model_load) / (1024 * 1024), 2)

    print(f"Mem after model load: {mem_after_model_load}")

    images = prepare_data(args.batch)

    with tf.device("/GPU:0"):
        for _ in range(35):
            infer(model=model, images=images, batch=args.batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", choices=["oxford_flowers102", "cifar100"])
    parser.add_argument("--model-name")
    parser.add_argument("--weights")
    parser.add_argument("--batch", choices=[1, 16, 32], default=1, type=int)

    args = parser.parse_args()
    main(args)
