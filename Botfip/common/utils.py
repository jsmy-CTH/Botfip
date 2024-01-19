import yaml
from omegaconf import OmegaConf
import torch
from torch import nn
from einops import rearrange, repeat
import signal
import numpy as np
import random

def list2str(orilist):
    orilist = [str(i) for i in orilist]
    return ','.join(orilist)

def str2list(str):
    new_list = str.split(',')
    return [int(i) for i in new_list]

def str2list_float(str):
    new_list = str.split(',')
    return [float(i) for i in new_list]

def config_create(model_config_yaml,key=None):
    with open(model_config_yaml, 'r') as f:
        model_yaml_config_dict = yaml.safe_load(f)
    model_config_dict = {}
    if key is not None:
        model_yaml_config_dict = model_yaml_config_dict[key]

    for k, v in model_yaml_config_dict.items():
        if isinstance(v,dict):
            for k1,v1 in v.items():
                model_config_dict[k1] = v1
        else:
            model_config_dict[k] = v
    return OmegaConf.create(model_config_dict)

def ind2points(funcimg,index):
    funcimg_values = []
    for i in range(index.size(0)):
        ind = index[i]
        ind = [(i,)+tuple(ind.tolist()) for i in range(funcimg.size(0))]
        funcimg_values_ind = torch.stack([funcimg[i] for i in ind],dim=0)
        funcimg_values.append(funcimg_values_ind)
    return torch.stack(funcimg_values,dim=1)

def initialize_weights(model):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.zeros_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data,0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zeros_()

def input_normalization(input_tensor,nest_num = 1):
    input_tensor_shape = input_tensor.shape
    func_max = torch.max
    func_min = torch.min
    func_subtract = torch.subtract
    func_divide = torch.divide

    for i in range(nest_num):
        func_max = torch.func.vmap(func_max)
        func_min = torch.func.vmap(func_min)
    input_max = func_max(input_tensor)
    input_min = func_min(input_tensor)

    for i in range(nest_num,len(input_tensor_shape)):
        input_max = input_max.unsqueeze(-1)
        input_min = input_min.unsqueeze(-1)


    input_tensor = 2*input_tensor - input_max - input_min
    input_tensor = input_tensor/(input_max-input_min+1e-8)

    return input_tensor,input_max,input_min


def get_same_ind(batch):
    # 扩展批处理的维度以使用广播
    expanded_batch = batch.unsqueeze(1)  # shape becomes [B, 1, S]
    transposed_batch = batch.unsqueeze(0)  # shape becomes [1, B, S]

    # 使用广播计算所有序列之间的相等关系
    equal_tensor = expanded_batch == transposed_batch  # shape [B, B, S]

    # 确定哪些序列完全相等
    same_ind = equal_tensor.all(dim=-1).int()  # shape [B, B]

    # 取反，使得0代表相同，1代表不同
    same_ind = 1 - same_ind

    return same_ind



def generate_two_points(interval, distance_range):
    min_dist, max_dist = distance_range

    # 确定第二个点的可能范围
    point1 = np.random.uniform(interval[0], interval[1])
    lower_bound = max(point1 - max_dist, interval[0])
    upper_bound = min(point1 + max_dist, interval[1])

    # 如果有效范围小于最小距离，则重新选择第一个点
    while upper_bound - lower_bound < min_dist:
        point1 = np.random.uniform(interval[0], interval[1])
        lower_bound = max(point1 - max_dist, interval[0])
        upper_bound = min(point1 + max_dist, interval[1])

    point2 = np.random.uniform(lower_bound, upper_bound)

    # 再次确保点之间的距离满足给定的范围
    while not (min_dist <= abs(point2 - point1) <= max_dist):
        point2 = np.random.uniform(lower_bound, upper_bound)

    if point1 > point2:
        point1, point2 = point2, point1

    point1 = np.round(point1,2)
    point2 = np.round(point2,2)

    return [point1, point2]


def set_last_true_to_false(matrix):
    """
    将布尔矩阵中每一行的最后一个True值设置为False。

    参数:
    matrix (torch.Tensor): 布尔张量。

    返回:
    torch.Tensor: 更新后的张量。
    """
    # 确保输入是布尔张量
    if not matrix.dtype == torch.bool:
        raise ValueError("输入必须是布尔张量")

    # 将布尔张量转换为整数张量，以便使用torch.argmax
    int_matrix = matrix.int()

    # 找到每行最后一个True的索引位置
    last_trues = torch.argmax(int_matrix.flip(dims=[1]), dim=1)

    # 创建行的索引
    row_indices = torch.arange(matrix.size(0)).to(matrix.device)

    # 过滤掉全False行
    mask = matrix.any(dim=1)
    row_indices = row_indices[mask]
    last_trues = last_trues[mask]

    # 更新矩阵中的相应位置
    matrix[row_indices, matrix.size(1) - 1 - last_trues] = False

    return matrix

class TimeoutError(Exception):
    def __init__(self, msg):
        super(TimeoutError, self).__init__()
        self.msg = msg


def time_out(interval, callback):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("run func timeout")

        def wrapper(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(interval)  # interval秒后向进程发送SIGALRM信号
                result = func(*args, **kwargs)
                signal.alarm(0)  # 函数在规定时间执行完后关闭alarm闹钟
                return result
            except TimeoutError as e:
                callback(e)
                return None

        return wrapper

    return decorator


def timeout_callback(e):
    print(e.msg)


def sample_without_replacement(lst, sample_size):
    remaining = lst.copy()  # 复制列表，用于跟踪剩余的可选元素
    sampled = []  # 跟踪已采样的元素

    while len(remaining) >= sample_size:
        # 从剩余元素中采样
        selection = random.sample(remaining, sample_size)
        sampled.extend(selection)

        # 更新剩余元素列表
        remaining = [item for item in remaining if item not in selection]

    # 如果剩余元素不足以满足采样大小，从已采样元素中补充
    if remaining:
        required = sample_size - len(remaining)
        sampled.extend(remaining)
        remaining = [item for item in sampled if item not in remaining]
        sampled.extend(random.sample(remaining, required))

    return sampled


def grouped_sample_without_replacement(lst, sample_size):
    remaining = lst.copy()  # 复制列表，用于跟踪剩余的可选元素
    sampled_groups = []  # 保存按组分开的采样结果

    while len(remaining) >= sample_size:
        # 从剩余元素中采样
        selection = random.sample(remaining, sample_size)
        sampled_groups.append(selection)

        # 更新剩余元素列表
        remaining = [item for item in remaining if item not in selection]

    # 如果剩余元素不足以满足采样大小，从已采样元素中补充
    if remaining:
        required = sample_size - len(remaining)
        extra_sample = random.sample(remaining + [item for group in sampled_groups for item in group], required)
        sampled_groups.append(remaining + extra_sample)

    return sampled_groups