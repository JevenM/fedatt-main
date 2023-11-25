#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'E:/Doctor1/coding/FGL_FrameWork/data'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# 使用L2距离，距离大的给与大权重
def aggregate_att(w_clients, w_server, stepsize):
    import copy
    import torch
    import torch.nn.functional as F
    # from scipy import linalg
    # import numpy as np
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k]-w_clients[i][k], ord=2)))
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize)
    return w_next

# ×基于consine相似度，相似度高的给与大权重
def aggregate_att2(w_clients, w_server, stepsize):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 将张量展平为二维张量
            flat_w_server = w_server[k].view(-1)
            flat_w_clients = w_clients[i][k].view(-1)
            # 计算余弦相似度
            att[k][i] = F.cosine_similarity(flat_w_server, flat_w_clients, dim=-1)
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize)
    return w_next

# ×基于consine相似度，直接依据相似度大小作为权重聚合模型各个key
def aggregate_att3(w_clients, w_server, stepsize):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 将张量展平为二维张量
            flat_w_server = w_server[k].view(-1)
            flat_w_clients = w_clients[i][k].view(-1)
            # 计算余弦相似度
            att[k][i] = F.cosine_similarity(flat_w_server, flat_w_clients, dim=-1)
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_clients[i][k], att[k][i])
        w_next[k] = torch.mul(att_weight, stepsize)
    return w_next

# 基于consine相似度，相似度高的赋予较低权重，根据权重直接聚合模型各个key的差距，在参考全局模型的预测性能来进行更新
def aggregate_att4(w_clients, w_server, global_model_acc):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 将张量展平为二维张量
            flat_w_server = w_server[k].view(-1)
            flat_w_clients = w_clients[i][k].view(-1)
            # 计算余弦相似度
            att[k][i] = F.cosine_similarity(flat_w_server, flat_w_clients, dim=-1)
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax((1-att[k])/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, (1-global_model_acc))
    return w_next


# 基于consine相似度，相似度高的赋予较低权重，根据权重直接聚合模型各个key的差距，再参考全局模型的预测性能来进行更新，
# 最终模型需要考虑采样率，如果采样率很低，那么权重也会受影响
# TODO 还会受到轮次的影响，一开始的模型肯定不准，性能也比较低，所以如何将此因素也考虑进去？
def aggregate_att5(w_clients, w_server, global_model_acc, sample_rate):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = average_weights(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 将张量展平为二维张量
            flat_w_server = w_server[k].view(-1)
            flat_w_clients = w_clients[i][k].view(-1)
            # 计算余弦相似度
            att[k][i] = F.cosine_similarity(flat_w_server, flat_w_clients, dim=-1)
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax((1-att[k])/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, (1-global_model_acc))
    
    for k in w_next.keys():
        global_next[k] = sample_rate*w_next[k] + (1-sample_rate)*avg_g_w[k]

    return global_next


# ×基于L2距离算相似度，效果最差，加上ratio
def aggregate_att6(w_clients, w_server, global_model_acc, sample_rate, ratio):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = average_weights(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 计算余弦相似度
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, (1-global_model_acc)*ratio)
    
    for k in w_next.keys():
        global_next[k] = sample_rate*w_next[k] + (1-sample_rate)*avg_g_w[k]

    return global_next

# 还是L2距离算相似度，去掉ratio
def aggregate_att7(w_clients, w_server, global_model_acc, sample_rate, ratio):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = average_weights(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1-global_model_acc)
    
    for k in w_next.keys():
        global_next[k] = sample_rate*w_next[k] + (1-sample_rate)*avg_g_w[k]

    return global_next

# ×还是L2距离算相似度，去掉sample_rate，替换为ratio，效果最差
def aggregate_att8(w_clients, w_server, global_model_acc, sample_rate, ratio):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = average_weights(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            # 计算余弦相似度
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1-global_model_acc)
    
    for k in w_next.keys():
        global_next[k] = ratio*w_next[k] + (1-ratio)*avg_g_w[k]

    return global_next

# 还是L2距离算相似度
def aggregate_att9(w_clients, w_server, global_model_acc):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1-global_model_acc)

    return w_next

# 还是L2距离算相似度，去掉global_model_acc
def aggregate_att10(w_clients, w_server):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1)

    return w_next

# 还是L2距离算相似度，加上平均模型融合
def aggregate_att11(w_clients, w_server, ratio):
    import copy
    import torch
    import torch.nn.functional as F
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    avg_g_w = average_weights(w_clients)
    global_next = copy.deepcopy(w_server)
    w_next = copy.deepcopy(w_server)
    att = {}
    # 初始化，归零
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm(w_server[k].cpu() - w_clients[i][k].cpu(), p=2).item()
    for k in w_next.keys():
        # 计算张量的标准差
        std = F.sigmoid(1/torch.std(att[k]))
        att[k] = F.softmax(att[k]/std, dim=0)
    for k in w_next.keys():
        # w_server[k].shape = att_weight.shape: torch.size([6,3,5,5])
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            # att[k][i]是个值，w_server[k]-w_clients[i][k]: [6,3,5,5]
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, 1)
    for k in w_next.keys():
        global_next[k] = ratio*w_next[k] + (1-ratio)*avg_g_w[k]
    return w_next

def exp_details(args, log):
    log.info('\nExperimental details:')
    log.info(f'    Model     : {args.model}')
    log.info(f'    Optimizer : {args.optimizer}')
    log.info(f'    Learning  : {args.lr}')
    log.info(f'    Global Rounds   : {args.epochs}\n')

    log.info('    Federated parameters:')
    if args.iid:
        log.info('    IID')
    else:
        log.info('    Non-IID')
    log.info(f'    Fraction of users  : {args.frac}')
    log.info(f'    Local Batch size   : {args.local_bs}')
    log.info(f'    Local Epochs       : {args.local_ep}\n')



def set_logger(log_file_path="D:/Git/fedatt-main/logs/", file_name=""):
    import logging

    # 创建一个 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置控制台日志级别

    # 创建一个处理器，用于将日志输出到文件

    file_handler = logging.FileHandler(log_file_path+file_name)
    file_handler.setLevel(logging.DEBUG)  # 设置文件日志级别

    # 创建一个格式化器，用于设置日志的格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 示例日志
    # logger.debug('This is a debug message')
    # logger.info('This is an info message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    # logger.critical('This is a critical message')
    return logger

def set_seed(args):
    '''
        set global seed
    '''

    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    # 禁用 cuDNN 自动调整策略，并启用确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True