import torch
import yaml
from collections import namedtuple
import pynvml
from pprint import pprint
def get_cuda_id(thre = 1) -> int: #
    pynvml.nvmlInit()
    """
    get idle cuda id which memory reserved upper than 1G
    Args:
        thre: thresold for gpu idle

    Returns:

    """
    for cuda_id in range(torch.cuda.device_count()):
        if pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(cuda_id)).used / 1e9 > thre:
            continue
        else:
            return cuda_id

def load_yaml(yaml_path='Config/ENP_default_s_512_c_150.yaml'):
    with open(yaml_path, 'r') as f:
        args = yaml.full_load(f.read())
    Args = namedtuple('ArgTuple', args.keys())
    return Args(**args), args



