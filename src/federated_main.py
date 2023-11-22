#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from datetime import datetime
import os
import copy
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, aggregate_att, set_logger, set_seed


if __name__ == '__main__':

    start_time = datetime.now()

    # define paths
    # path_project = os.path.abspath('..')
    # print(path_project) # D:/Git

    args = args_parser()
    set_seed(args)
    event_dir_name = '{}_{}_{}_C{}_iid{}_E{}_B{}'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    comments = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+event_dir_name
    # 默认runs下
    logger = SummaryWriter(comment=event_dir_name)
    log = set_logger(log_file_path='logs/', file_name=comments+".log")
    exp_details(args, log)
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda:'+ str(args.gpu) if args.gpu!=-1 else 'cpu'
    log.info(device)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    log.info(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        log.info(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device=device, log=log)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        if args.agg == 'avg':
            global_weights = average_weights(local_weights)
        elif args.agg == 'att':
            global_weights = aggregate_att(local_weights, copy.deepcopy(global_model).state_dict(), 1)


        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device=device, log=log)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            log.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
            log.info(f'Training Loss : {np.mean(np.array(train_loss))}')
            log.info('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset, device)

    log.info(f' \n Results after {args.epochs} global rounds of training:')
    log.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    log.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    end_time = datetime.now()
    h_, remainder_ = divmod((end_time - start_time).seconds, 3600)
    m_, s_ = divmod(remainder_, 60)
    time_str_ = "Time %02d:%02d:%02d" % (h_, m_, s_)
    log.info(f'\n Total Run {time_str_}')

    import os
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 获取上一级目录
    parent_directory = os.path.abspath(os.path.join(current_dir, os.pardir))

    # Saving the objects train_loss and train_accuracy:
    # 使用相对路径拼接文件路径
    file_path = 'save/objects/{}_{}_{}_C{}_iid{}_E{}_B{}_agg{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.agg)

    file_name = os.path.join(parent_directory, file_path)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    # 从 .pkl 文件中读取数据
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)

    # 打印读取的数据
    log.info(loaded_data)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    loss_file_name = os.path.join(parent_directory, 'save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_agg{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.agg))
    plt.savefig(loss_file_name)
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    acc_file_name = os.path.join(parent_directory, 'save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_agg{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.agg))
    plt.savefig(acc_file_name)
