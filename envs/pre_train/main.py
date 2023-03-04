#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
import dataset
from utils import check_overfitting, early_stop, logger
from train import train
from metric import ACC, AUC
from test import test
import loss
from itertools import product
import time
from tensorboardX import SummaryWriter
import yaml
from dataset import train_dataset
from CDM import * 

def main():

    CONFIG = yaml.load(open('./config.yml', 'r'), Loader=yaml.Loader)
    CONFIG_DATA = yaml.load(open('../../data/{}/info_filtered.yml'.format(CONFIG['dataset_name']), 'r'), Loader=yaml.Loader)

    #  set env
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cuda')

    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #  load data
    train_data = train_dataset(CONFIG['path'], CONFIG['dataset_name'], CONFIG_DATA['kc_maxid']+1, 'train')
    val_data = train_dataset(CONFIG['path'], CONFIG['dataset_name'], CONFIG_DATA['kc_maxid']+1, 'val')

    train_loader = DataLoader(train_data, batch_size=32, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=8, pin_memory=True)

    #  metric
    metrics = [ACC(), AUC()]
    TARGET = 'AUC'

    #  loss
    loss_func = loss.BCELoss('sum')
    # name
    model_name = CONFIG['model']
    #  log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'], 
        f"{model_name}", ''), 'best', checkpoint_target=TARGET)
 
    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

    lr = CONFIG[model_name]['lr']

    # model
    if 'NCD' in model_name:
        info = NCD_Info(model_name, CONFIG[model_name]['layers_fc_dim'], CONFIG[model_name]['layers_fc_dropout'])
        model = NCD(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1).to(device)
        print(model)
    elif 'MIRT' in model_name:
        info = MIRT_Info(model_name, CONFIG[model_name]['dim'], CONFIG[model_name]['guess'])
        model = MIRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1).to(device)
        print(model)
    elif 'IRT' in model_name and "MIRT" not in model_name:
        info = IRT_Info(model_name, CONFIG[model_name]['guess'])
        model = IRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1).to(device)
        print(model)
    
    # op
    op = optim.Adam(model.parameters(), lr=lr)
    # env
    env = {'lr': lr,
            'op': str(op).split(' ')[0],   # Adam
            'dataset': CONFIG['dataset_name'],
            'model': model_name, 
            }

    # log
    log.update_modelinfo(info, env, metrics)

    # train & test
    early = CONFIG[model_name]['early']  
    for epoch in range(CONFIG[model_name]['epochs']):
        # train one ep
        trainloss = train(model, epoch+1, train_loader, op, device, CONFIG, loss_func)

        # test
        if epoch % CONFIG['test_interval'] == 0:  
            output_metrics = test(model, val_loader, device, CONFIG, metrics)
            # log
            log.update_log(metrics, model) 

            # check overfitting
            if epoch > 1:
                if check_overfitting(log.metrics_log, TARGET, 1, show=True):
                    break
            # early stop
            early = early_stop(log.metrics_log[TARGET], early, threshold=0)
            if early <= 0:
                print("early stop!!!")
                break

    log.close_log(TARGET)
    log.close()


if __name__ == "__main__":
    main()
