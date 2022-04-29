# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class SharedDropout_convex(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout_convex, self).__init__()

        self.p = p
        self.batch_first = batch_first
        self.mask = None

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.mask is not None: mask = self.mask
            else:
                if self.batch_first:
                    mask = self.get_mask(x[:, 0], self.p)
                else:
                    mask = self.get_mask(x[0], self.p)
                self.mask_fix = mask
                self.mask = mask
            #print(x.size())
            #print(mask.size())
            x = x * (mask.unsqueeze(1) if self.batch_first else mask)

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask

    def set(self,index):
        if index is None: 
            self.mask = None
            self.mask_fix = None
        else: 
            self.mask = self.mask_fix[index]


"""
class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, y, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))
            scale = 3.0 / (2.0 * x_mask + y_mask + eps)
            x_mask *= scale
            y_mask *= scale
            x *= x_mask.unsqueeze(dim=-1)
            y *= y_mask.unsqueeze(dim=-1)

        return x, y
"""
class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items


