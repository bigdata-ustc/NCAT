import torch
import torch.nn as nn

import torch.nn.functional as F
import scipy.sparse as sp 
import numpy as np
from .model_base import Info, Model
import loss
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import os
import yaml
from utils import tensor_to_numpy

class NCD_Info(Info):
    def __init__(self, name, layers_fc_dim, layers_fc_dropout):
        super().__init__(name)
        self.layers_fc_dim = layers_fc_dim
        self.layers_fc_dropout = layers_fc_dropout


class NCD(Model):
    '''
    NeuralCDM
    '''
    def __init__(self, info, num_users, exer_n, knowledge_n, load_path=None):
        super().__init__(info, num_users, exer_n, knowledge_n)

        self.device = torch.device('cuda')

        self.prednet_input_len = self.knowledge_n
        # network structure
        self.student_emb = nn.Embedding(self.num_users, self.knowledge_n)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_n)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # MLP
        prednet_full_list = []
        for i in range(len(info.layers_fc_dim)):
            if i == 0:
                prednet_full_list.append(nn.Linear(self.prednet_input_len, info.layers_fc_dim[i]))
            else:
                prednet_full_list.append(nn.Linear(info.layers_fc_dim[i-1], info.layers_fc_dim[i]))
            prednet_full_list.append(nn.Sigmoid())
            prednet_full_list.append(nn.Dropout(info.layers_fc_dropout[i]))
        self.prednet = nn.ModuleList(prednet_full_list)

        # output layer
        self.prednet_full_output = nn.Linear(info.layers_fc_dim[-1], 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        if load_path:
            self.load_model(load_path)
            for name, param in self.named_parameters():
                if 'student_emb' not in name:
                    param.requires_grad = False


    def update(self, cat_data, lr, epoch):
        self.train()
        cat_loader = DataLoader(cat_data, batch_size=100, num_workers=1, pin_memory=True)
        op = optim.Adam(params=filter(lambda p:p.requires_grad, self.parameters()), lr=lr)
        loss_func = loss.BCELoss(reduction='mean')
        for _ in range(epoch):
            for i, data in enumerate(cat_loader):
                op.zero_grad()
                user_id, exer_id, score, knowledge_emb = data
                # print(user_id, exer_id, score, knowledge_emb)
                modelout = self.forward(user_id.to(self.device), exer_id.to(self.device), knowledge_emb.to(self.device))
                model_loss = loss_func(modelout, score.float().to(self.device))
                model_loss.backward()
                op.step()

    
    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items() if 'student' not in k}
        self.load_state_dict(load_dict, strict=False)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        kn_emb = kn_emb.float()
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb

        for layer in self.prednet:
            input_x = layer(input_x)
        output = self.prednet_full_output(input_x)
        output = torch.sigmoid(output).view(-1)

        return output


    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data
    
    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.prednet:
            layer.apply(clipper)
        self.prednet_full_output.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
