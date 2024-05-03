import time
import subprocess
import logging
import os

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)


file_handler = logging.FileHandler('log.txt')
file_handler.setLevel(logging.INFO)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_list(path):
    files = os.listdir(path)
    return files

scripts_queue = get_list('ToBeRun')

import pynvml


def check_gpu_memory():

    pynvml.nvmlInit()


    gpu_count = pynvml.nvmlDeviceGetCount()


    gpu_memory_usage = {}

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        memory_usage = info.used / 1024 / 1024
        gpu_memory_usage[gpu_name] = memory_usage


    pynvml.nvmlShutdown()

    return gpu_memory_usage


def start_script(script_name):

    output_file = f'{script_name}.txt'
    cmd = f"cd /Projects/PSAS/ && /share/apps/anaconda3/envs/bin/python {script_name} > logs/{output_file}"
    subprocess.Popen(cmd, shell=True)
    logger.info(f'Script {script_name} has started.')


while True:
    gpu_memory_usage = check_gpu_memory()


    for gpu, memory in gpu_memory_usage.items():
        if memory < 1000:
            available_gpu = True
        else:
            available_gpu = False

    if available_gpu and len(scripts_queue)>0:
        script_to_start = scripts_queue[0]
        start_script(script_to_start)
        scripts_queue.pop(0)

    time.sleep(30)
