import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.utils import lazy_property

from torch.autograd import grad

from alg import inside, inside_ng, eisner, inside2, eisner2o




class DependencyCRF:

    #initialization
    def __init__(self, E, mE, E_sib, lengths, tmask, smask):
        if E is not None: self.E = E.clone()
        if mE is not None: self.mE = mE.clone()
        if E_sib is not None: self.E_sib = E_sib.clone()
        self.lengths = lengths
        if tmask.size(0)!=E.size(0): self.tmask = tmask.repeat(3,1,1)
        else: self.tmask = tmask
        #mask for batchify
        #self.idx = E.new_zeros(E.size()[0], E.size()[1])
        #for i in range(E.size()[0]):
            #self.idx[i,:lengths[i]+1] = 1
        #self.mask = (self.idx==1)
        self.mask = smask
        #self.idx[:,0] = 0
        self.mask1 = smask.clone()
        self.mask1[:,0] = False
        #self.mask1 = (self.idx==1)

    @lazy_property
    def partition(self):
        self.E.masked_fill_(~self.mask.unsqueeze(1), float('-inf'))
        if not self.E.requires_grad: self.E = self.E.detach().requires_grad_()
        if not self.E_sib.requires_grad: self.E_sib = self.E_sib.detach().requires_grad_()
        logz = inside2(scores=(self.E, self.E_sib), mask=self.mask1)
        return logz

    @lazy_property
    def gpartition(self):
        self.E.masked_fill_(~self.mask.unsqueeze(1), float('-inf'))
        self.E = self.E.detach().requires_grad_()
        self.E_sib = self.E_sib.detach().requires_grad_()
        logz  = inside2(scores=(self.E, self.E_sib), mask=self.mask1)
        return logz

    def log_prob(self, tree, arc, sib):
        E = (tree*self.mE).sum(dim=(-1,-2))
        arc_seq = arc[self.mask1]
        sib_seq = sib[self.mask1]
        sib_mask = sib_seq.gt(0)
        sib_seq = sib_seq[sib_mask]

        #get the new lengths of sib score
        nmask = self.mask1 * sib.gt(0)
        nlen = nmask.sum(-1)

        s_sib = self.E_sib[self.mask1][torch.arange(len(arc_seq)), arc_seq]
        s_sib = s_sib[sib_mask].gather(-1, sib_seq.unsqueeze(-1)).squeeze(-1)
        s_sib = pad_sequence(s_sib.split(nlen.tolist()),True)
        return E + s_sib.sum(-1) - self.partition.squeeze(0)
        #return E.sum() + s_sib.sum() - self.partition.sum()

    def prob(self):
        logZ = self.gpartition
        probs, = grad(logZ.sum(), self.E, retain_graph=False)
        return probs

    @lazy_property
    def argmax(self):
        with torch.no_grad():
            #print(self.E)
            self.E.diagonal(0, 1, 2).fill_(float('-inf'))
            preds = eisner(self.E, self.mask1)
            #print('predict of preds')
            #print(preds)
            cptree = preds.new_zeros(preds.size(0),preds.size(1),preds.size(1))
            cptree.scatter_(1, preds.unsqueeze(-2), 1)
            ptree = cptree[:,1:,1:]
            idx = torch.arange(ptree.size(1))
            ptree[:,idx,idx] = cptree[:,0,1:]
            ptree[~self.tmask] = 0
            return ptree

    @lazy_property
    def argmax2o(self):
        with torch.no_grad():
            #print(self.E)
            #self.E.diagonal(0, 1, 2).fill_(float('-inf'))
            
            preds = eisner2o((self.E, self.E_sib),self.mask1)
            #print('predict of preds')
            #print(preds)
            cptree = preds.new_zeros(preds.size(0),preds.size(1),preds.size(1))
            cptree.scatter_(1, preds.unsqueeze(-2), 1)
            ptree = cptree[:,1:,1:]
            idx = torch.arange(ptree.size(1))
            ptree[:,idx,idx] = cptree[:,0,1:]
            ptree[~self.tmask] = 0
            return ptree, preds

    #to implement
    def topk(self):
        pass
