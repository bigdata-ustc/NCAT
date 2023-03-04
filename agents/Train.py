#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
import copy as cp
import logger
from util import *
import time
from collections import Counter
import copy as cp

MEMORYSIZE = 50000
BATCHSIZE = 128
THRESHOLD = 300

start = 0
end = 3000

def decay_function1(x):
    x = 50+x
    return max(2.0/(1+np.power(x,0.2)),0.001)

START = decay_function1(start)
END = decay_function1(end)

def decay_function(x):
    x = max(min(end,x),start)
    return (decay_function1(x)-END)/(START-END+0.0000001)


class Train(object):
    def __init__(self, env, fa, args):
        self.env = env
        self.fa = fa
        self.args = args
        self.tau = 0
        self.memory = []

    def train(self):
        for epoch in range(self.args.training_epoch):
            logger.log(epoch)
            self.collecting_data_update_model("training", epoch)
            if epoch % 100 == 0 and epoch >= 300:
                self.collecting_data_update_model("validation", epoch)
                self.collecting_data_update_model("evaluation", epoch)
            self.env.re_split_data()

    def collecting_data_update_model(self, type="training", epoch=0):
        if type=="training":
            selected_users = np.random.choice(self.env.training,(self.args.inner_epoch,))
        elif type=="validation":
            selected_users = self.env.validation
        elif type=="evaluation":
            selected_users = self.env.evaluation
        elif type=="verified":
            selected_users =  self.env.training
        else:
            selected_users = range(1,3)
        infos = {item:[] for item in self.args.ST}
        used_actions = []
        for uuid in selected_users:
            actions = {}
            rwds = 0
            done = False
            state = self.env.reset_with_users(uuid)
            while not done:
                data = {"uid": [state[0][1]]}
                for i in range(2):
                    p_r, pnt = self.convert_item_seq2matrix([[0]+[item[0] for item in state[1] if item[3]["rate"] == i]])
                    data["p_"+str(i)+"_rec"] = p_r
                    data["p_"+str(i)+"_t"] = pnt
                policy = self.fa.predict(data)[0]
                if type == "training":
                    if np.random.random()<5*THRESHOLD/(THRESHOLD+self.tau): policy = np.random.uniform(0,1,(self.args.item_num,))
                for item in actions: policy[item] = -np.inf
                for item in range(self.args.item_num):
                    if item not in self.env.candidate_items:
                        policy[item] = -np.inf
                action = np.argmax(policy[1:]) + 1
                s_pre = cp.deepcopy(state)
                state_next, rwd, done, info = self.env.step(action)
                if type == "training":
                    self.memory.append([s_pre,action,rwd,done,cp.deepcopy(state_next)])
                actions[action] = 1
                rwds += rwd
                state = state_next
                if len(state[1]) in self.args.ST:
                    infos[len(state[1])].append(info)
            used_actions.extend(list(actions.keys()))
        if type == "training":
            if len(self.memory) >= self.args.batch:
                self.memory = self.memory[-MEMORYSIZE:]
                batch = [self.memory[item] for item in np.random.choice(range(len(self.memory)),(self.args.batch,))]
                data = self.convert_batch2dict(batch,epoch)
                loss = self.fa.optimize_model(data, self.args.learning_rate)
                logger.record_tabular("loss ", "|".join([str(np.round(loss, 4))]))
                self.tau += 5
        # if type != 'training':
        for item in self.args.ST:
            logger.record_tabular(str(item)+"ACC",round(np.mean([i["ACC"] for i in infos[item]]),4))
            logger.record_tabular(str(item)+"AUC",round(np.mean([i["AUC"] for i in infos[item]]),4))
            logger.log(str(item)+" ACC: ",round(np.mean([i["ACC"] for i in infos[item]]),4))
        logger.record_tabular("epoch", epoch)
        logger.record_tabular("type", type)
        logger.dump_tabular()

    def convert_batch2dict(self, batch, epoch):
        uids = []
        pos_recs = {i:[] for i in range(2)}
        next_pos = {i:[] for i in range(2)}
        iids = []
        goals = []
        dones = []
        for item in batch:
            uids.append(item[0][0][1])
            ep = item[0][1] # 
            for xxx in range(2):
                pos_recs[xxx].append([0] + [j[0] for j in ep if j[3]["rate"]==xxx])
            iids.append(item[1]) # action  
            goals.append(item[2])
            if item[3]:dones.append(0.0)
            else:dones.append(1.0)
            ep = item[4][1] 
            for xxx in range(2):
                next_pos[xxx].append([0] + [j[0] for j in ep if j[3]["rate"] == xxx])
        data = {"uid":uids}
        for xxx in range(2):
            p_r, pnt = self.convert_item_seq2matrix(next_pos[xxx])
            data["p_" + str(xxx) + "_rec"] = p_r
            data["p_" + str(xxx) + "_t"] = pnt

        value = self.fa.predict(data)
        value[:,0] = -500
        goals = np.max(value,axis=-1)*np.asarray(dones)*min(self.args.gamma,decay_function(max(end-epoch,0)+1)) + goals
        data = {"uid":uids,"iid":iids,"goal":goals}
        for i in range(2):
            p_r, pnt = self.convert_item_seq2matrix(pos_recs[i])
            data["p_" + str(i) + "_rec"] = p_r
            data["p_" + str(i) + "_t"] = pnt
        return data

    def convert_item_seq2matrix(self, item_seq):
        max_length = max([len(item) for item in item_seq])
        matrix = np.zeros((len(item_seq), max_length), dtype=np.int32)
        for x, xx in enumerate(item_seq):
            for y, yy in enumerate(xx):
                matrix[x, y] = yy
        target_index = [len(i) - 1 for i in item_seq]

        return matrix, target_index
