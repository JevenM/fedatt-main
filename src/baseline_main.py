#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset, set_logger, set_seed
from options import args_parser
from update import test_inference
from models import MLP, CNNCifar100, CNNMnist, CNNFashion_Mnist, CNNCifar
from datetime import datetime

if __name__ == '__main__':
    start_time = datetime.now()
    args = args_parser()
    set_seed(args)
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    device = 'cuda:'+ str(args.gpu) if args.gpu!=-1 else 'cpu'
    event_dir_name = 'nn{}_{}_{}'.\
        format(args.dataset, args.model, args.epochs)

    comments = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+event_dir_name
    log = set_logger(log_file_path='logs/', file_name=comments+".log")
    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

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

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=args.wd)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        log.info(f'Train loss: {loss_avg}')
        epoch_loss.append(loss_avg)

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset, device)
    log.info(f'Results after {args.epochs} rounds of training:')
    log.info(f'|---- Test on {len(test_dataset)} samples')
    log.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
    end_time = datetime.now()
    h_, remainder_ = divmod((end_time - start_time).seconds, 3600)
    m_, s_ = divmod(remainder_, 60)
    time_str_ = "Time %02d:%02d:%02d" % (h_, m_, s_)
    log.info(f'Total Run {time_str_}')


    import os
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # src
    # print(current_dir)
    # 获取上一级目录
    parent_directory = os.path.abspath(os.path.join(current_dir, os.pardir))
    # D:\Git\fedatt-main
    # print(parent_directory)
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    loss_file_name = os.path.join(parent_directory, 'save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
    plt.savefig(loss_file_name)
    log.info(epoch_loss)
    print(comments)