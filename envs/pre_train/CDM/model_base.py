#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Info(object):
    '''
    [FOR `utils.logger`]

    the base class that packing all hyperparameters and infos used in the related model
    '''

    def __init__(self, name):
        self.name = name


    def get_title(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(lambda x: dct[x].get_title() if isinstance(dct[x], Info) else x, dct.keys()))

    def get_csv_title(self):
        return self.get_title().replace('\t', ', ')

    def __getitem__(self, key):
        if hasattr(self, '_info'):
            return self._info[key]
        else:
            return self.__getattribute__(key)

    def __str__(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(str, dct.values()))

    def get_line(self):
        return self.__str__()

    def get_csv_line(self):
        return self.get_line().replace('\t', ', ')

class Model(nn.Module):
    '''
    base class for all MF-based model
    packing embedding initialization, embedding choosing in forward

    NEED IMPLEMENT:
    - `propagate`: all raw embeddings -> processed embeddings(user/bundle)
    - `predict`: processed embeddings of targets(users/bundles inputs) -> scores

    OPTIONAL:
    - `regularize`: processed embeddings of targets(users/bundles inputs) -> extra loss(default: L2)
    - `get_infotype`: the correct type of `info`(default: `object`)
    '''

    def get_infotype(self):
        return object

    def __init__(self, info, num_users, exer_n, knowledge_n):
        super().__init__()
        assert isinstance(info, self.get_infotype())
        self.info = info
        self.num_users = num_users
        self.exer_n = exer_n
        self.knowledge_n = knowledge_n

    def update(self, cat_data, lr):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def predict(self, stu_ids, topredict, knowledge_map):
        self.eval()
        input_stu_id = []
        input_exer_id = []
        input_kn_emb = []
        for idx, stu in enumerate(stu_ids):
            exer_ids = np.argwhere(topredict[idx] == 1).reshape(-1)
            input_exer_id.extend(exer_ids)
            input_stu_id.extend([stu] * len(exer_ids))
            for exer_i in exer_ids:
                knowledge_emb = [0.] * self.knowledge_n
                for knowledge_code in knowledge_map[exer_i]:
                    knowledge_emb[knowledge_code - 1] = 1.0
                input_kn_emb.append(knowledge_emb)
        input_stu_id, input_exer_id, input_kn_emb = torch.LongTensor(input_stu_id), torch.LongTensor(input_exer_id), torch.FloatTensor(input_kn_emb)
        pred = self.forward(input_stu_id.to(self.device), input_exer_id.to(self.device), input_kn_emb.to(self.device))

        return pred, input_stu_id.to(self.device), input_exer_id.to(self.device)





if __name__ == "__main__":
    info = Info(5,2)
    print(info.get_title())
