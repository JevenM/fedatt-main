#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
from models import MLP, CNNCifar100, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import aggregate_att16, aggregate_att11, aggregate_att10, aggregate_att12, aggregate_att13, \
    aggregate_att14, aggregate_att15, aggregate_att17, aggregate_att18, aggregate_att19, aggregate_att2, aggregate_att20, aggregate_att3, aggregate_att4, \
aggregate_att5, aggregate_att6, aggregate_att7, aggregate_att8, aggregate_att9, get_dataset, average_weights, exp_details, \
aggregate_att, set_logger, set_seed


if __name__ == '__main__':

    start_time = datetime.now()

    # define paths
    # path_project = os.path.abspath('..')
    # print(path_project) # D:/Git

    args = args_parser()
    set_seed(args)
    # 20240103之前的所有学习率都是0.01
    event_dir_name = '{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.lr,
               args.local_ep, args.local_bs, args.agg)

    comments = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+event_dir_name
    # 默认runs下
    logger = SummaryWriter(comment=event_dir_name)
    log = set_logger(log_file_path='logs/', file_name=comments+".log")
    exp_details(args, log)
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    log.info(f"is cuda available? {torch.cuda.is_available()}")
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
        elif args.dataset == 'cifar10':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            global_model = CNNCifar100(args=args)

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
    val_loss_list, val_acc_list = [], []
    print_every = 2
    best_test_acc = 0.0
    global_model_acc = 0.0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # local_models = []
        log.info(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        log.info(f"selected users: {idxs_users}")
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device=device, log=log, client_id=idx)
            w, loss, l_model = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            # local_models.append(copy.deepcopy(l_model))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        if args.agg == 'avg':
            global_weights = average_weights(local_weights)
        elif args.agg == 'att':
            global_weights = aggregate_att(local_weights, copy.deepcopy(global_model).state_dict(), 1, args.dataset)
        elif args.agg == 'att2':
            global_weights = aggregate_att2(local_weights, copy.deepcopy(global_model).state_dict(), 1)
            # global_weights = aggregate_att2(local_models, copy.deepcopy(global_model), 1)
        elif args.agg == 'att3':
            global_weights = aggregate_att3(local_weights, copy.deepcopy(global_model).state_dict(), 1)
        elif args.agg == 'att4':
            global_weights = aggregate_att4(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc)
        elif args.agg == 'att5':
            global_weights = aggregate_att5(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.frac)
        elif args.agg == 'att6':
            global_weights = aggregate_att6(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.frac, (epoch+1)/args.epochs)
        elif args.agg == 'att7':
            global_weights = aggregate_att7(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.frac, (epoch+1)/args.epochs)
        elif args.agg == 'att8':
            global_weights = aggregate_att8(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.frac, (epoch+1)/args.epochs)
        elif args.agg == 'att9':
            global_weights = aggregate_att9(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc)
        elif args.agg == 'att10':
            global_weights = aggregate_att10(local_weights, copy.deepcopy(global_model).state_dict())
        elif args.agg == 'att11':
            global_weights = aggregate_att11(local_weights, copy.deepcopy(global_model).state_dict(), (epoch+1)/args.epochs)
        elif args.agg == 'att12':
            global_weights = aggregate_att12(local_weights, copy.deepcopy(global_model).state_dict(), (epoch+1)/args.epochs)
        elif args.agg == 'att13':
            global_weights = aggregate_att13(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc)
        elif args.agg == 'att14':
            global_weights = aggregate_att14(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.dataset)
        elif args.agg == 'att15':
            global_weights = aggregate_att15(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc)
        elif args.agg == 'att16':
            global_weights = aggregate_att16(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, (epoch+1)/args.epochs, args.dataset)
        elif args.agg == 'att17':
            global_weights = aggregate_att17(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, args.dataset)
        elif args.agg == 'att18':
            global_weights = aggregate_att18(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, (epoch+1)/args.epochs)
        elif args.agg == 'att19':
            global_weights = aggregate_att19(local_weights, copy.deepcopy(global_model).state_dict(), (epoch+1)/args.epochs)
        elif args.agg == 'att20':
            global_weights = aggregate_att20(local_weights, copy.deepcopy(global_model).state_dict(), global_model_acc, (epoch+1)/args.epochs, args.dataset)
        
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger, device=device, log=log, client_id=c)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        global_model_acc = train_accuracy[-1]

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            log.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
            log.info(f'Training Loss : {np.mean(np.array(train_loss))}')
            log.info('Train Accuracy: {:.2f}% \n'.format(100*global_model_acc))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset, device)
        val_loss_list.append(test_loss)
        val_acc_list.append(test_acc)
        log.info(f' \n Results after {args.epochs} global rounds of training:')
        log.info("|---- Avg Train Accuracy: {:.2f}%".format(100*global_model_acc))
        log.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            log.info("|---- New Best Test Accuracy: {:.2f}%".format(100*test_acc))

    end_time = datetime.now()
    h_, remainder_ = divmod((end_time - start_time).seconds, 3600)
    m_, s_ = divmod(remainder_, 60)
    time_str_ = "Time %02d:%02d:%02d" % (h_, m_, s_)
    log.info(f'\n Total Run {time_str_}')
    log.info("|---- Best Test Accuracy: {:.2f}%".format(100*best_test_acc))

    import os
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 获取上一级目录
    parent_directory = os.path.abspath(os.path.join(current_dir, os.pardir))

    # Saving the objects train_loss and train_accuracy:
    # 使用相对路径拼接文件路径
    file_path = 'save/objects/new/{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.lr,
               args.local_ep, args.local_bs, args.agg)

    file_name = os.path.join(parent_directory, file_path)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, val_loss_list, val_acc_list], f)

    # 从 .pkl 文件中读取数据
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)

    # 打印读取的数据
    log.info(loaded_data)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    x_label = 'Communication Rounds'
    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel(x_label)
    loss_file_name = os.path.join(parent_directory, 'save/newimg/fed_{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}_loss_train.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.lr, args.local_ep, args.local_bs, args.agg))
    plt.savefig(loss_file_name)
    
    # Plot Average Train Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Train Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel(x_label)
    acc_file_name = os.path.join(parent_directory, 'save/newimg/fed_{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}_acc_train.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.lr, args.local_ep, args.local_bs, args.agg))
    plt.savefig(acc_file_name)

    # Plot LOSS curve
    plt.figure()
    plt.title('Test loss vs Communication rounds')
    plt.plot(range(len(val_loss_list)), val_loss_list, color='g')
    plt.ylabel('Testing loss')
    plt.xlabel(x_label)
    loss_file_name_t = os.path.join(parent_directory, 'save/newimg/fed_{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}_loss_test.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.lr, args.local_ep, args.local_bs, args.agg))
    plt.savefig(loss_file_name_t)
    
    # Plot Test Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(val_acc_list)), val_acc_list, color='b')
    plt.ylabel('Test Accuracy')
    plt.xlabel(x_label)
    acc_file_name_t = os.path.join(parent_directory, 'save/newimg/fed_{}_{}_{}_C{}_iid{}_lr{}_E{}_B{}_agg{}_acc_test.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.lr, args.local_ep, args.local_bs, args.agg))
    plt.savefig(acc_file_name_t)
    print(comments)
