import torch
import torch.nn as nn
from torch.distributions.utils import lazy_property

class DependencyCE:

    #initialization
    def __init__(self, E, lengths):
        self.E = E
        self.lengths = lengths
        #mask for batchify
        self.mask = E.new_zeros(E.size())
        self.mask1 = E.new_zeros(E.size()[0], E.size()[1])
        for i in range(E.size()[0]):
            self.mask[i,:lengths[i],:lengths[i]] = 1
            self.mask1[i,:lengths[i]] = 1
        self.idx = (self.mask==1)
        self.idx1 = (self.mask1==1)
    
    @lazy_property
    def partition(self):
        E = self.E.clone()
        E[~self.idx] = -1e34
        maxE = E.max(1)[0]
        tE = (torch.exp(E-maxE.unsqueeze(1))*self.mask).sum(1)
        tE[~self.idx1] = 1
        maxE[~self.idx1] = 0
        Z = maxE + torch.log(tE)
        return Z.sum(1)

    def log_prob(self, tree):
        E = (tree*self.E).sum(dim=(-1,-2))
        return E-self.partition

    #to implement
    def topk(self):
        pass

