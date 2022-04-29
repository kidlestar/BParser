import torch
import torch.nn as nn

from torch.distributions.utils import lazy_property

from alg import inside, inside_ng


import torch
import torch.nn as nn

from torch.distributions.utils import lazy_property

from torch.autograd import grad

from alg import inside, inside_ng, eisner, mst




class DependencyCRF:

    #initialization
    def __init__(self, E, mE, lengths, tmask):
        self.E = E
        self.mE = mE
        self.lengths = lengths
        if tmask.size(0)!=E.size(0): self.tmask = tmask.repeat(3,1,1)
        else: self.tmask = tmask
        #mask for batchify
        self.idx = E.new_zeros(E.size()[0], E.size()[1])
        for i in range(E.size()[0]):
            self.idx[i,:lengths[i]+1] = 1
        self.mask = (self.idx==1)
        self.idx[:,0] = 0
        self.mask1 = (self.idx==1)

    @lazy_property
    def partition(self):
        self.E.masked_fill_(~self.mask.unsqueeze(1), float('-inf'))
        if self.E.requires_grad: s_i, s_c = inside(self.E, self.mask1)
        else: s_i, s_c = inside_ng(self.E, self.mask1)
        return s_c[0].gather(0, (self.lengths).unsqueeze(0))

    @lazy_property
    def gpartition(self):
        self.E.masked_fill_(~self.mask.unsqueeze(1), float('-inf'))
        self.E = self.E.detach().requires_grad_()
        s_i, s_c = inside(self.E, self.mask1)
        return s_c[0].gather(0, (self.lengths).unsqueeze(0)).sum()

    def log_prob(self, tree):
        E = (tree*self.mE).sum(dim=(-1,-2))
        return E-self.partition

    def prob(self):
        logZ = self.gpartition
        probs, = grad(logZ, self.E, retain_graph=False)
        return probs

    @lazy_property
    def argmax(self):
        with torch.no_grad():
            #print(self.E)
            E = self.E.detach().clone()
            E.diagonal(0, 1, 2).fill_(float('-inf'))
            preds = mst(E, self.mask1)
            #preds = eisner(E, self.mask1)
            #print('predict of preds')
            #print(preds)
            cptree = preds.new_zeros(preds.size(0),preds.size(1),preds.size(1))
            cptree.scatter_(1, preds.unsqueeze(-2), 1)
            ptree = cptree[:,1:,1:]
            idx = torch.arange(ptree.size(1))
            ptree[:,idx,idx] = cptree[:,0,1:]
            ptree[~self.tmask] = 0
            return ptree

    #to implement
    def topk(self):
        pass
"""


class DependencyCRF:

    #initialization
    def __init__(self, E, mE, lengths):
        self.E = E
        self.mE = mE
        self.lengths = lengths
        #mask for batchify
        self.idx = E.new_zeros(E.size()[0], E.size()[1])
        for i in range(E.size()[0]):
            self.idx[i,:lengths[i]+1] = 1
        self.mask = (self.idx==1)
        self.idx[:,0] = 0
        self.mask1 = (self.idx==1)

    @lazy_property
    def partition(self):
        self.E.masked_fill_(~self.mask.unsqueeze(1), float('-inf'))
        if self.E.requires_grad: s_i, s_c = inside(self.E, self.mask1)
        else: s_i, s_c = inside_ng(self.E, self.mask1)
        return s_c[0].gather(0, (self.lengths).unsqueeze(0))

    def log_prob(self, tree):
        E = (tree*self.mE).sum(dim=(-1,-2))
        return E-self.partition

    #to implement
    def topk(self):
        pass

"""
