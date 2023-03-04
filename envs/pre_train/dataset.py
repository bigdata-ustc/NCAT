#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import scipy.sparse as sp 
import yaml
from datetime import datetime
import json
import random
import sys
from torchvision import transforms
from  torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F
import scipy.sparse as sp 
import torch.optim as optim
from collections import defaultdict
from collections import Counter
from CDM import *
from utils import *
import tqdm


class train_dataset(Dataset):
    def __init__(self, path, name, knowledge_n, task, knowledge_map):
        assert task in ['train', 'val']
        self.task = task
        self.path = path
        self.name = name
        self.data = []
        
        if path is not None:
            data_file = os.path.join(self.path, self.name, '{}_set.json'.format(self.task))
            with open(data_file, encoding='utf8') as i_f:
                self.data = json.load(i_f)

        self.knowledge_dim = knowledge_n
        self.knowledge_map = knowledge_map
  
    def __getitem__(self, idx):
        """
        """
        set_info = self.data[idx]
        user_id = set_info['user_id']
        exer_id = set_info['exer_id']
        score = set_info['score']
        knowledge_emb = [0.] * self.knowledge_dim
        for knowledge_code in set_info['knowledge_code']:
            knowledge_emb[knowledge_code - 1] = 1.0
        return user_id, exer_id, score, np.array(knowledge_emb)


    def __len__(self):
        return len(self.data)

    
    def add_record(self, user_index, item_index, correct):
        for i in range(len(user_index)):
            # user_index, item_index, correct = record
            one_record = {}
            one_record['user_id'] = user_index[i]
            one_record['exer_id'] = item_index[i]
            one_record['score'] = correct[i]
            one_record['knowledge_code'] = self.knowledge_map[item_index[i]]
            self.data.append(one_record)
        
    def clear(self):
        self.data = []

    