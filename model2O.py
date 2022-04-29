import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import time
from multiprocessing import Pool
import pickle
import time

from torch.distributions.gumbel import Gumbel

import decoder
import utils
from DependencyCRF import DependencyCRF
from DependencyCE import DependencyCE
from sparsemax import Sparsemax

#model for train, predict and evaluate
class model:

    #initialization
    def __init__(self, device, optimizer, scheduler, net):

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.sparsemax = Sparsemax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    
    #predict the trees which minimize the modified energy of network (AdaAFW)
    def evaluate(self, sent, bsent, char, pos, batch_size, lengths, mask, mask1, mask2):
        
        #do once initialization
        self.net.eval()
        self.net.EVAInitialization(sent, bsent, char, pos, batch_size, lengths, mask, mask1, mask2)
        

        #E_arc = self.net.linearMatrix
        E_rel = self.net.logRel.max(-1)[0]
        #E = E_arc + E_rel
        nE_rel = E_rel.new_zeros(E_rel.size(0), E_rel.size(1)+1, E_rel.size(2)+1)
        nE_rel[:,1:,1:] = E_rel.transpose(-1,-2)
        idxx = torch.arange(self.net.max_length)
        nE_rel[:,1:,0] = E_rel[:,idxx,idxx]
        prob = DependencyCRF(self.net.E, self.net.linearMatrix, self.net.E_sib, self.net.lengths,self.net.mask, mask2).prob()
        #new decoding method
        nE = torch.log(prob + 1e-45) + nE_rel
        x, pred = DependencyCRF(self.net.E+nE_rel, self.net.linearMatrix+E_rel, self.net.E_sib, self.net.lengths,self.net.mask,mask2).argmax2o
        x_ = DependencyCRF(nE, None, None, self.net.lengths,self.net.mask,mask2).argmax
        rel = self.net.Erel.detach().argmax(-1)
        

        return x, x_, rel

    def get_sibs(self, sequence):
        sibs = [-1] * (len(sequence) + 1)
        heads = [0] + [int(i) for i in sequence]

        for i in range(1, len(heads)):
            hi = heads[i]
            for j in range(i + 1, len(heads)):
                hj = heads[j]
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i] = j
                    else:
                        sibs[j] = i
                    break
        return sibs

    def a2s(self, arc, batch_size, lengths):
        list_sib = []
        for i in range(batch_size):
            seq = arc[i][1:].tolist()
            seq = seq[:lengths[i].cpu()]
            sib = self.get_sibs(seq)
            list_sib.append(torch.tensor(sib))
        sib = pad_sequence(list_sib,batch_first=True) .to(self.device)
        return sib
        
    def get_arc(self, tree, mask):
        smask = mask.clone()
        smask[:,0] = False
        etree = tree.new_zeros(tree.size(0),tree.size(1)+1, tree.size(2)+1)
        etree[:,1:,1:] = tree.transpose(-1,-2)
        etree[:,1:,0] = tree[:,idxx,idxx]
        s = etree.nonzero(as_tuple=False)
        arc = tree.zeros(tree.size(0),tree.size(1)+1)
        arc.masked_scatter_(smask,s[:,2])
        return arc


    def NetEnergy(self, tree, arc, sib, mE, E_sib, mask, isa=False):
        E = (tree*mE).sum(dim=(-1,-2))
        arc_seq = arc[mask]
        sib_seq = sib[mask]
        sib_mask = sib_seq.gt(0)
        sib_seq = sib_seq[sib_mask]

        #get the new lengths of sib score
        nmask = mask * sib.gt(0)
        nlen = nmask.sum(-1)

        s_sib = E_sib[mask][torch.arange(len(arc_seq)), arc_seq]
        s_sib = s_sib[sib_mask].gather(-1, sib_seq.unsqueeze(-1)).squeeze(-1)
        s_sib = pad_sequence(s_sib.split(nlen.tolist()),True)
        if isa: return (E, s_sib.sum(-1))
        else: return E + s_sib.sum(-1)


    #train for the input (well formed)
    def train(self, sent, bsent, char, pos, deprel, tree, sib, arc, mask, mask1, mask2, batch_size, lengths):
        self.net.train()
        self.net.Initialization(sent, bsent, char, pos, deprel, tree, batch_size, lengths, mask, mask1, mask2)
       
        
        idxx = torch.arange(tree.size(-1))

        etree = tree.new_zeros(tree.size(0),tree.size(1)+1, tree.size(2)+1)
        etree[:,1:,1:] = tree.transpose(-1,-2)
        etree[:,1:,0] = tree[:,idxx,idxx]


        deptree = DependencyCRF(self.net.E-2*etree, self.net.linearMatrix-2*tree, self.net.E_sib, self.net.lengths,self.net.mask,mask2)
        
        ptree, parc = deptree.argmax2o
        psib = self.a2s(parc, batch_size, lengths)
        tmask = mask2.clone()
        tmask[:,0] = False
        E = self.NetEnergy(ptree, parc, psib, self.net.linearMatrix, self.net.E_sib, tmask)
        
        TEl, TEg = self.NetEnergy(tree, arc, sib, self.net.linearMatrix, self.net.E_sib, tmask, isa=True)
        TE = TEl + TEg
        print('linear energy: ' + str(torch.abs(TEl).sum().item()))
        print('global energy: ' + str(torch.abs(TEg).sum().item()))

        loss = self.relu(E-TE+((1-ptree)*tree + (1-tree)*ptree).sum(dim=(-1,-2))).sum()
        
        #deptree = DependencyCRF(self.net.E, self.net.linearMatrix, self.net.E_sib, self.net.lengths,self.net.mask,mask2)
        #loss = - deptree.log_prob(self.net.tree, arc, sib).sum()
            
        logRel = self.net.logRel
        logRel = logRel.transpose(1,2)[self.net.tree.transpose(-1,-2)==1]
            
        ERel = self.net.Erel
        ERel = ERel.transpose(1,2)[self.net.tree.transpose(-1,-2)==1]

            
        lossRel = self.criterion(ERel, self.net.deprel[self.net.mask1])
            
        
        print('loss over arc: ' + str((loss/(self.net.lengths.sum())).item()))
        print('loss over label: ' + str((lossRel/(self.net.lengths.sum())).item()))
        meanLoss = (loss+lossRel)/(self.net.lengths.sum())

        meanLoss.backward()
        norm = nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        print('value of norm: ' + str(norm.item()))
        self.optimizer.step()
        self.scheduler.step()
        ret = meanLoss.item()
        del loss
        del meanLoss
        
        return ret

