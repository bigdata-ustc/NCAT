#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    log_interval = CONFIG['log_interval']
    model_name = CONFIG['model']
    model.train()
    start = time()
    for i, data in enumerate(loader):
        optim.zero_grad()
        user_id, exer_id, score, knowledge_emb = data
        modelout = model(user_id.to(device), exer_id.to(device), knowledge_emb.to(device))
        loss = loss_func(modelout, score.float().to(device))
        loss.backward()
        optim.step()
        if model_name == 'NCD':
            model.apply_clipper()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                100. * (i+1) / len(loader), loss))
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss



