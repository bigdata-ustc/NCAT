#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import tensor_to_numpy
_is_hit_cache = {}


def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1),
                              col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit


class _Metric:
    '''
    base class of metrics like Recall@k NDCG@k MRR@k
    '''

    def __init__(self):
        self.start()

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, ground_truth):
        '''
        - scores: model output
        - ground_truth: one-hot test dataset shape=(users, all_bundles/all_items).
        '''
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        '''
        clear all
        '''
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0
        self._preds = []
        self._labels = []

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum/self._cnt


class ACC(_Metric):
    '''
    accuracy
    '''
    def __init__(self):
        super().__init__()
        self.thresh = 0.5 

    def get_title(self):
        return "ACC"

    def __call__(self, scores, ground_truth):
        scores = tensor_to_numpy(scores)
        ground_truth = tensor_to_numpy(ground_truth)
        scores = np.where(scores > self.thresh, 1, 0)
        self._cnt += len(ground_truth)
        self._sum += np.sum(np.equal(scores, ground_truth)) 

class right_rate(_Metric):
    '''
    right_rate
    '''
    def __init__(self):
        super().__init__()
        self.thresh = 0.5 

    def get_title(self):
        return "right_rate"

    def __call__(self, scores, ground_truth):
        scores = tensor_to_numpy(scores)
        ground_truth = tensor_to_numpy(ground_truth)
        scores = np.where(scores > self.thresh, 1, 0)
        self._cnt += len(scores)
        self._sum += np.sum(scores) 





class AUC(_Metric):
    '''
    AUC
    '''
    def __init__(self):
        super().__init__()

    def get_title(self):
        return "AUC"

    def __call__(self, scores, ground_truth):
        scores = tensor_to_numpy(scores)
        ground_truth = tensor_to_numpy(ground_truth)
        self._preds.extend(scores)
        self._labels.extend(ground_truth)
    
    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = roc_auc_score(np.array(self._labels).astype(np.int16), self._preds)


class Recall(_Metric):
    '''
    Recall in top-k samples
    '''

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epison = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        is_hit = get_is_hit(scores, ground_truth, self.topk) 
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item() 
        self._sum += (is_hit/(num_pos+self.epison)).sum().item()

