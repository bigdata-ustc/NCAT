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
# from utils import *
from utils import tensor_to_numpy
import random

class IRT_Info(Info):
    def __init__(self, name, guess):
        super().__init__(name)
        self.guess = guess

class IRT(Model):
    '''
    IRT
    '''
    def __init__(self, info, num_users, exer_n, knowledge_n, load_path=None):
        super().__init__(info, num_users, exer_n, knowledge_n)
        self.device = torch.device('cuda')
        self.thetas = nn.Embedding(num_users, 1)
        self.alphas = nn.Embedding(exer_n, 1)
        self.betas = nn.Embedding(exer_n, 1)
        self.guess = nn.Embedding(exer_n, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        if load_path:
            self.load_model(load_path)
            for name, param in self.named_parameters():
                if 'thetas' not in name:
                    param.requires_grad = False
    
    def init_stu_emb(self):
        for name, param in self.named_parameters():
            if 'thetas' in name and 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, exer_id, kn_emb):
        theta = self.thetas(stu_id)
        alpha = self.alphas(exer_id)
        beta = self.betas(exer_id)
        guess = torch.sigmoid(self.guess(exer_id))
        pred = torch.sigmoid(alpha * (theta - beta))   

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

    def cal_loss(self, user_index, item_index, correct, knowledge_map):
        self.eval()
        with torch.no_grad():
            kn_emb = []
            for exer_i in item_index:
                knowledge_emb = [0.] * self.knowledge_n
                for knowledge_code in knowledge_map[exer_i]:
                    knowledge_emb[knowledge_code - 1] = 1.0
                kn_emb.append(knowledge_emb)
            user_index, item_index, correct, kn_emb = torch.LongTensor(user_index), torch.LongTensor(item_index), torch.FloatTensor(correct), torch.FloatTensor(kn_emb)
        
            loss_func = loss.BCELoss(reduction='mean')
            modelout = self.forward(user_index.to(self.device), item_index.to(self.device), kn_emb.to(self.device))
            model_loss = loss_func(modelout, correct.to(self.device))
        return tensor_to_numpy(model_loss), tensor_to_numpy(modelout)




    



    @staticmethod
    def irf(alpha, beta, theta, guess=None):
        """ item response function
        pred = guess + (1 - guess) * pred
        """
        x = np.array(alpha*(theta - beta)).astype(np.float128)
        prob2 = 1.0 / (1.0 + np.exp(-x))
        # print(np.exp(-alpha*(theta - beta)))
        if guess is not None:
            guess = 1.0 / (1.0 + np.exp(-1 * guess))
            return guess + (1-guess) * prob2
        else:
            return prob2

    @staticmethod
    def pd_irf_alpha(alpha, beta, theta, guess):
        """partial derivative of item response function to alpha
        """
        guess = 1.0 / (1.0 + np.exp(-1 * guess))
        p = IRT.irf(alpha, beta, theta)
        q = 1 - p
        return (1-guess)* p * q * (theta - beta)

    @staticmethod
    def pd_irf_beta(alpha, beta, theta, guess):
        """ partial derivative of item response function to beta
        """
        guess = 1.0 / (1.0 + np.exp(-1 * guess))
        p = IRT.irf(alpha, beta, theta)
        q = 1 - p
        return (1-guess)* p * q * (-alpha)

    @staticmethod
    def pd_irf_theta(alpha, beta, theta, guess):
        """ partial derivative of item response function to theta
        """
        guess = 1.0 / (1.0 + np.exp(-1 * guess))
        p = IRT.irf(alpha, beta, theta)
        q = 1 - p
        return (1-guess) * p * q * alpha

    def _theta_params(self, stu_ids):
        device = self.thetas.weight.device
        theta = self.thetas(torch.LongTensor(stu_ids).to(device))
        return tensor_to_numpy(theta.view(-1))

    def _alpha_params(self, exer_ids):
        device = self.alphas.weight.device
        alpha = self.alphas(torch.LongTensor(exer_ids).to(device))
        return tensor_to_numpy(alpha.view(-1))

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
        load_dict = {k:v for k, v in state_dict.items() if 'theta' not in k}
        self.load_state_dict(load_dict, strict=False)
