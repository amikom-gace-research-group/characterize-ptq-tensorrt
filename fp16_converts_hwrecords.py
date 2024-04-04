import re
import os
import time
from datetime import datetime 
import subprocess
import psutil
import threading
import argparse

class CPU(threading.Thread):
    def __init__(self, pid, task_completed):
        super().__init__()
        self.pid = pid
        self.task_completed = task_completed
        self.stop_event = threading.Event()
        self.cpu_usage_data = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get the process object using the PID
                process = psutil.Process(self.pid)

                # Check if the process is running
                if not process.is_running():
                    break

                output = subprocess.check_output([
                    'pidstat', '-p', str(self.pid), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                self.cpu_usage_data.append(cpu_)

            except psutil.NoSuchProcess:
                # Process no longer exists, stop monitoring
                break
            except Exception as e:
                print("An error occurred:", e)

            # Check if the task has completed
            if self.task_completed.is_set():
                break
        
        self.stop_monitoring()

    def stop_monitoring(self):
        # Set the stop event to stop hardware monitoring
        self.stop_event.set()


class Memory(threading.Thread):
    def __init__(self, pid, task_completed):
        super().__init__()
        self.pid = pid
        self.task_completed = task_completed
        self.stop_event = threading.Event()
        self.mem_usage_data = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get the process object using the PID
                process = psutil.Process(self.pid)

                # Check if the process is running
                if not process.is_running():
                    break

                output = subprocess.check_output([
                    'pidstat', '-p', str(self.pid), '1', '1', '-r'])
                mem = float(output.splitlines()[-2].split()[-3])
                self.mem_usage_data.append(mem)

            except psutil.NoSuchProcess:
                # Process no longer exists, stop monitoring
                break
            except Exception as e:
                print("An error occurred:", e)

            # Check if the task has completed
            if self.task_completed.is_set():
                break
        
        self.stop_monitoring()

    def stop_monitoring(self):
        # Set the stop event to stop hardware monitoring
        self.stop_event.set()


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
        result = sum(entire) / len(entire)
    except:
        result = 0
        entire = entire
        pass

    subprocess.check_output("rm test.txt", shell=True)
    return result, entire


# Example usage
def start_process(command):
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    return process.pid


def calculate_average(data):
    # print(data)
    return sum(data) / len(data)


def main(args):
    os.system("echo 'CloudLab12#$%' | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")

    # Warmup
    command = [
            "polygraphy",
            "convert",
            f"generated/onnx/{args.cfg}-{args.model_name}-1.onnx",
            "--fp16",
            "--output",
            f"generated/trt/fp16-{args.cfg}-{args.model_name}-1.engine"
    ]
    pid = start_process(command)
    process = psutil.Process(pid)
    process.wait()
    
    print("Warmup done")

    for i in range(1, 6):
        # Start subprocess
        command = [
            "polygraphy",
            "convert",
            f"generated/onnx/{args.cfg}-{args.model_name}-{i}.onnx",
            "--fp16",
            "--output",
            f"generated/trt/fp16-{args.cfg}-{args.model_name}-{i}.engine"
        ]
        begin = datetime.now()
        pid = start_process(command)

        # Create event to signal task completion
        task_completed = threading.Event()

        # Start monitoring thread
        cpu_monitor = CPU(pid, task_completed)
        mem_monitor = Memory(pid, task_completed)
        cpu_monitor.start()
        mem_monitor.start()
        _jstat_start()

        # Wait for the subprocess to complete
        process = psutil.Process(pid)
        process.wait()
        end = datetime.now()

        elapsed = (end - begin).total_seconds()

        # Signal task completion
        task_completed.set()

        # Wait for the monitoring thread to finish
        cpu_monitor.join()
        mem_monitor.join()

        avg_gpu_use = _jstat_stop()[0]
        avg_cpu_use = calculate_average(cpu_monitor.cpu_usage_data)
        avg_mem_use = calculate_average(mem_monitor.mem_usage_data)

        print(f"{round(elapsed, 2)},{round(avg_cpu_use, 2)},{round(avg_gpu_use, 2)},{round((avg_mem_use / 1024), 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name')
    parser.add_argument('--cfg')
    
    args = parser.parse_args()
    main(args)
