#!/usr/bin/python
# encoding: utf-8
import sys
sys.path.append('./envs/pre_train')
import numpy as np
import logger
from collections import OrderedDict
from util import *
import math
from collections import Counter
import copy as cp
import json
from .pre_train.CDM import *
from .pre_train.dataset import train_dataset
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# TODO:
class env(object):
    def __init__(self, args):
        logger.log("initialize environment")
        self.T = args.T
        self.data_name = args.data_name
        self.CDM = args.CDM
        self.rates = {}
        self.users = {}
        self.utypes = {}
        self.args = args
        self.device = torch.device('cuda')
        self.rates, self._item_num, self.know_map = self.load_data(os.path.join(self.args.data_path, args.data_name, "log_data_filtered.json"))
        logger.log("user number: " + str(len(self.rates) + 1)) 
        logger.log("item number: " + str(self._item_num + 1))
        self.setup_train_test()
        self.sup_rates, self.query_rates = self.split_data(ratio=0.5)
        
        print('loading CDM %s' % args.CDM)
        self.model, self.dataset = self.load_CDM()
        print(self.model)
    
    def split_data(self, ratio=0.5):
        sup_rates, query_rates = {}, {}
        for u in self.rates:
            all_items = list(self.rates[u].keys())
            np.random.shuffle(all_items)
            sup_rates[u] = {it: self.rates[u][it] for it in all_items[:int(ratio*len(all_items))]}
            query_rates[u] = {it: self.rates[u][it] for it in all_items[int(ratio*len(all_items)):]}
        return sup_rates, query_rates

    def re_split_data(self, ratio=0.5):
        self.sup_rates, self.query_rates = self.split_data(ratio)

    @property
    def candidate_items(self):
        return set(self.sup_rates[self.state[0][0]].keys())

    @property
    def user_num(self):
        return len(self.rates) + 1

    @property
    def item_num(self):
        return self._item_num + 1

    @property
    def utype_num(self):
        return len(self.utypes) + 1

    def load_CDM(self):
        name = self.CDM
        CONFIG = yaml.load(open('./envs/pre_train/config.yml', 'r'), Loader=yaml.Loader)
        CONFIG_DATA = yaml.load(open('./data/{}/info_filtered.yml'.format(self.data_name), 'r'), Loader=yaml.Loader)
        cat_data = train_dataset(None, self.args.data_name, CONFIG_DATA['kc_maxid']+1, 'train', self.know_map)
        # cat_loader = DataLoader(cat_data, batch_size=32, num_workers=1, pin_memory=True, shuffle=True)

        best_model_path = './envs/model_file/{}/{}'.format(self.data_name, CONFIG[name]['best_model_path'])
        
        if 'NCD' in name:
            info = NCD_Info(name, CONFIG[name]['layers_fc_dim'], CONFIG[name]['layers_fc_dropout'])
            model = NCD(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)

        elif 'MIRT' in name:
            self.ismirt = True
            info = MIRT_Info(name, CONFIG[name]['dim'], CONFIG[name]['guess'])
            model = MIRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)

        elif 'IRT' in name and 'MIRT' not in name:
            info = IRT_Info(name, CONFIG[name]['guess'])
            model = IRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)
        return model, cat_data

    def setup_train_test(self):
        users = list(range(1, self.user_num))
        np.random.shuffle(users)
        self.training, self.validation, self.evaluation = np.split(np.asarray(users), [int(.8 * self.user_num - 1),
                                                                                       int(.9 * self.user_num - 1)])

    def load_data(self, path):
        
        with open(path, encoding='utf8') as i_f:
            stus = json.load(i_f) # list
        rates = {}
        items = set()
        user_cnt = 0
        know_map = {}

        for stu in stus:
            if stu['log_num'] < self.T * 2:
                continue
            user_cnt += 1
            rates[user_cnt] = {}
            for log in stu['logs']: 
                rates[user_cnt][int(log['exer_id'])] = int(log['score'])
                items.add(int(log['exer_id']))
                know_map[int(log['exer_id'])] = log['knowledge_code']

        max_itemid = max(items)
        
        return rates, max_itemid, know_map

    def reset(self):
        self.reset_with_users(np.random.choice(self.training))

    def reset_with_users(self, uid):
        self.state = [(uid,1), []]
        self.short = {}
        return self.state

    def step(self, action):
        assert action in self.sup_rates[self.state[0][0]] and action not in self.short
        reward, ACC, AUC, rate = self.reward(action)

        if len(self.state[1]) < self.T - 1:
            done = False
        else:
            done = True

        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        info = {"ACC": ACC,
                "AUC": AUC,
                "rate":rate}
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info

    def reward(self, action):

        self.dataset.clear()
        items = [state[0] for state in self.state[1]] + [action]
        correct = [self.rates[self.state[0][0]][it] for it in items]
        self.dataset.add_record([self.state[0][0]]*len(items), items, correct)
        self.model.update(self.dataset, self.args.learning_rate, epoch=1)

        item_query = list(self.query_rates[self.state[0][0]].keys())
        correct_query = [self.rates[self.state[0][0]][it] for it in item_query]
        loss, pred = self.model.cal_loss([self.state[0][0]]*len(item_query), item_query, correct_query, self.know_map)
        # ACC AUC
        pred_bin = np.where(pred > 0.5, 1, 0)
        ACC = np.sum(np.equal(pred_bin, correct_query)) / len(pred_bin) 
        try:
            AUC = roc_auc_score(correct_query, pred)
        except ValueError:
            AUC = -1
        self.model.init_stu_emb()
        return -loss, ACC, AUC, correct[-1]



    def precision(self, episode):
        return sum([i[1] for i in episode])

    

    def recall(self, episode, uid):
        return sum([i[1] for i in episode]) / len(self.rates[uid])

    def step_policy(self,policy):
        policy = policy[:self.args.T]
        rewards = []
        for action in policy:
            if action in self.rates[self.state[0][0]]:
                rewards.append(self.rates[self.state[0][0]][action])
            else:
                rewards.append(0)
        t = [[a,rewards[i],False] for i,a in enumerate(policy)]
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0])}
        self.state[1].extend(t)
        return self.state,rewards,True,info



    def ndcg(self, episode, uid):
        if len(self.rates[uid]) > len(episode):
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(sorted(list(self.rates[uid].values()),reverse=True),
                                                           len(episode),
                                                           method=1)
        else:
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(
                list(self.rates[uid].values()) + [0] * (len(episode) - len(self.rates[uid])),
                len(episode), method=1)

    def dcg_at_k(self, r, k, method=1):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')

    def alpha_dcg(self, item_list, k=10, alpha=0.5, *args):
        items = []
        G = []
        for i, item in enumerate(item_list[:k]):
            items += item
            G.append(sum(map(lambda x: math.pow(alpha, x - 1), dict(Counter(items)).values())) / math.log(i + 2, 2))
        return sum(G)

if __name__ == '__main__':
    args = {'T':10, 'data_path': './data/data/'}
    env(args)
