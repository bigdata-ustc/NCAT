#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        for user_id, exer_id, score, knowledge_emb in loader:
            modelout = model(user_id.to(device), exer_id.to(device), knowledge_emb.to(device))
            for metric in metrics:
                metric(modelout, score.to(device))
                
    print('Test: time={:d}s'.format(int(time()-start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    return metrics
