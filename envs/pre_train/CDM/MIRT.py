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

class MIRT_Info(Info):
    def __init__(self, name, dim, guess):
        super().__init__(name)
        self.dim = dim
        self.guess = guess

class MIRT(Model):
    '''
    MIRT
    '''
    def __init__(self, info, num_users, exer_n, knowledge_n, load_path=None):
        super().__init__(info, num_users, exer_n, knowledge_n)
        self.device = torch.device('cuda')
        self.K = self.info.dim
        self.thetas = nn.Embedding(num_users, self.K)
        self.alphas = nn.Embedding(exer_n, self.K)
        self.betas = nn.Embedding(exer_n, 1)
        self.guess = nn.Embedding(exer_n, 1)
        self.knowledge_layer = nn.Linear(knowledge_n, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        if load_path:
            self.load_model(load_path)
            for name, param in self.named_parameters():
                if 'thetas' not in name:
                    param.requires_grad = False


    def forward(self, stu_id, exer_id, kn_emb):
        theta = self.thetas(stu_id)
        alpha = self.alphas(exer_id)
        betas = self.betas(exer_id)
        guess = torch.sigmoid(self.guess(exer_id))
        pred = torch.sigmoid((alpha * theta).sum(dim=1, keepdim=True) - betas)   

        if self.guess:
            pred = guess + (1 - guess) * pred
        pred = pred.view(-1) 
        return pred
    
    
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

        
    @staticmethod
    def irf(alpha, beta, theta, guess=None):
        prob2 = 1.0 / (1.0 + np.exp(-(np.sum(alpha*theta, 1) - beta)))
        if guess is not None:
            guess = 1.0 / (1.0 + np.exp(-1 * guess))
            return guess + (1-guess) * prob2
        else:
            return prob2


    def _theta_params(self, stu_ids):
        device = self.thetas.weight.device
        theta = self.thetas(torch.LongTensor(stu_ids).to(device))
        return tensor_to_numpy(theta)

    def _alpha_params(self, exer_ids):
        device = self.alphas.weight.device
        alpha = self.alphas(torch.LongTensor(exer_ids).to(device))
        return tensor_to_numpy(alpha)

    def _beta_params(self, exer_ids):
        device = self.betas.weight.device
        beta = self.betas(torch.LongTensor(exer_ids).to(device))
        return tensor_to_numpy(beta.view(-1))

    def _guess_params(self, exer_ids):
        device = self.guess.weight.device
        guess = self.guess(torch.LongTensor(exer_ids).to(device))
        return tensor_to_numpy(guess.view(-1))

    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k:v for k, v in state_dict.items() if 'thetas' not in k}
        self.load_state_dict(load_dict, strict=False)
