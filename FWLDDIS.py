import torch
import torch.nn as nn

import numpy as np
import time

import calgorithm
import utils
import copy

def Eisner(input_data):
    (energy,length) = input_data
    gmatrix = energy[0:length,0:length]
    ngmatrix = np.zeros((length+1,length+1))
    predict = np.zeros(energy.shape)
    idx = np.arange(length)
    ngmatrix[0,idx+1] = gmatrix[idx,idx]
    ngmatrix[1:,1:] = gmatrix
    head = calgorithm.parse_proj(ngmatrix)
    for i in range(length):
        if head[i+1]==0: predict[i,i] = 1
        else: predict[head[i+1]-1,i] = 1
    return predict


#Lagevin dynamic with FW in the discrete domain, the gradient functions to approach the smallest point
class LDDISFW:

    def __init__(self, k=10):

        self.k = k

    def update(self, function):
        self.function = function
        self.batch_size = self.function.batch_size
        self.lengths = self.function.lengths
        self.max_length = self.lengths.max()
        self.device = self.function.device
        (matrix,lengths) = function.linearMatrix()
        map_inputs = [(matrix[i].cpu().numpy(),lengths[i].item()) for i in range(self.batch_size)]
        outputs = torch.tensor(np.stack(list(map(Eisner,map_inputs))), dtype=torch.float, device=self.device)
        self.x = [outputs[i,0:self.lengths[i],0:self.lengths[i]].requires_grad_(True) for i in range(self.batch_size)]

    def step(self, x=None):
        if x is not None:
            self.x = x
        for i in range(self.k):
            self.v = self.function.value(self.x)
            #print('energy of every iteration: ')
            #print(self.v.detach())
            #print('average energy: ' + str(self.v.detach().sum().item()))
            self.grad = self.function.grad()
            self.padgrad = utils.padding_structure(self.grad,self.batch_size,self.device,self.max_length)
            tmap_inputs = [(-self.padgrad[i].cpu().numpy(),self.lengths[i].item()) for i in range(self.batch_size)]
            toutputs = torch.tensor(np.stack(list(map(Eisner,tmap_inputs))), dtype=torch.float, device=self.device)
            #generate the gaussian noise
            self.noise = torch.randn(self.padgrad.size(), device=self.device)
            #add noise to gradient
            self.padgrad += 1.414*self.noise
            map_inputs = [(-self.padgrad[i].cpu().numpy(),self.lengths[i].item()) for i in range(self.batch_size)]
            outputs = torch.tensor(np.stack(list(map(Eisner,map_inputs))), dtype=torch.float, device=self.device)
            #print('difference: ' + str(torch.abs(toutputs-outputs).sum(dim=(1,2))/2.0))
            #print('length: ' + str(self.lengths))
            self.x = [outputs[i,0:self.lengths[i],0:self.lengths[i]].requires_grad_(True) for i in range(self.batch_size)]
        #exit()
        return self.v
        #return utils.padding_structure(self.x,self.batch_size,self.device,self.max_length).detach()




