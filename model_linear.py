import torch
import torch.nn as nn

import numpy as np
import time
from multiprocessing import Pool
import pickle
import time

import decoder
import utils
from FWFAC import AdaFW, simpleEisner, Eisner
#from FWLD import AdaFW, simpleEisner, Eisner
#from FWLDDIS import LDDISFW
#from sampling import MHSampling, IMSampling
from DependencyCRF import DependencyCRF
from DependencyCE import DependencyCE
from KEisner import EisnerK
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
        

        E_rel = self.net.logRel.max(-1)[0]
        nE_rel = E_rel.new_zeros(E_rel.size(0), E_rel.size(1)+1, E_rel.size(2)+1)
        nE_rel[:,1:,1:] = E_rel.transpose(-1,-2)
        idxx = torch.arange(self.net.max_length)
        nE_rel[:,1:,0] = E_rel[:,idxx,idxx]
        prob = DependencyCRF(self.net.E, self.net.linearMatrix, self.net.lengths,self.net.mask).prob()
        #new decoding method
        nE = torch.log(prob + 1e-45) + nE_rel
        x = DependencyCRF(prob, None, self.net.lengths,self.net.mask).argmax
        x_ = DependencyCRF(nE, None, self.net.lengths,self.net.mask).argmax
        rel = self.net.Erel.detach().argmax(-1)

        
        return x, x_, rel

    #train for the input (well formed)
    def train(self, sent, bsent, char, pos, deprel, tree, mask, mask1, mask2, batch_size, lengths):
        self.net.train()
        #print('in training')
        self.net.Initialization(sent, bsent, char, pos, deprel, tree, batch_size, lengths, mask, mask1, mask2)
        
        etree = tree.new_zeros(tree.size(0),tree.size(1)+1, tree.size(2)+1)
        idx = torch.arange(self.net.max_length)
        etree[:,1:,1:] = tree.transpose(-1,-2)
        etree[:,1:,0] = tree[:,idx,idx]

        #deptree = DependencyCRF(self.net.E-2*etree, self.net.linearMatrix-2*tree, self.net.lengths,self.net.mask)
        deptree = DependencyCE(self.net.linearMatrix-2*tree, self.net.lengths)
        
        loss = - deptree.log_prob(self.net.tree)
            
        ERel = self.net.Erel
        ERel = ERel.transpose(1,2)[self.net.tree.transpose(-1,-2)==1]

            
        lossRel = self.criterion(ERel, self.net.deprel[self.net.mask1])
        print('loss over arc: ' + str((loss.sum()/(self.net.lengths.sum())).item()))
        print('loss over label: ' + str((lossRel/(self.net.lengths.sum())).item()))
        meanLoss = (loss.sum()+lossRel)/(self.net.lengths.sum())

        meanLoss.backward()
        norm = nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        print('value of norm: ' + str(norm.item()))
        self.optimizer.step()
        self.scheduler.step()
        ret = meanLoss.item()
        del loss
        del meanLoss
        
        return ret

