#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. 
        `none`: no reduction will be applied, 
        `mean`: the sum of the output will be divided by the number of elements in the output, 
        `sum`: the output will be summed. 

        Note: size_average and reduce are in the process of being deprecated, 
        and in the meantime,  specifying either of those two args will override reduction. 
        Default: `sum`
        '''
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction
    



class BCELoss(_Loss):
    def __init__(self, reduction):
        super().__init__(reduction)
        self._loss_function = nn.BCELoss(reduction=self.reduction)

    def forward(self, preds, labels, **kwargs):
        loss = self._loss_function(preds, labels)
        return loss

