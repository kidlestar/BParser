import torch
import torch.nn as nn

import numpy as np
import time

import calgorithm
import utils
from multiprocessing import Pool
import sys
import copy

"""
def amax2(input_data):
    (energy,length) = input_data
    gmatrix = energy[0:length,0:length]
    predict = np.zeros(energy.shape)
"""    

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


class simpleEisner:

    def __init__(self, device, function):
        self.device = device
        self.function = function
        
    def update(self,function):
        self.function = function
        self.batch_size = self.function.batch_size
        self.lengths = self.function.lengths
        input_tree = [torch.zeros(self.lengths[i],self.lengths[i],requires_grad=True, device=self.device) for i in range(self.batch_size)]
        grad = self.function.grad(input_tree)
        #input_tree1 = [torch.ones(self.lengths[i],self.lengths[i],requires_grad=True, device=self.device) for i in range(self.batch_size)]
        #grad1 = self.function.grad(input_tree1)
        #print([torch.abs(grad[i]-grad1[i]).sum() for i in range(self.batch_size)])
        #exit()
        matrix = -utils.padding_structure(grad,self.batch_size,self.device,self.function.net.max_length)
        #(matrix,lengths) = function.linearMatrix()
        map_inputs = [(matrix[i].cpu().numpy(),self.lengths[i].item()) for i in range(self.batch_size)]
        outputs = np.stack(list(map(Eisner,map_inputs)))
        outputs = torch.tensor(outputs,dtype=torch.float,device=self.device)
        return outputs


     


#AdaFW(simple version) AdaAFW(away-step) AdaPFW(pairwise)
class AdaFW:

    def __init__(self, device, function, tolerance=1e-6, tau=2, eta=1.001, version='AdaFW'):

        self.device = device
        self.function = function
        self.tolerance = tolerance
        self.tau = tau
        self.eta = eta

        #multiprocessing for eisner
        #self.pool = Pool(20)

        self.version = version

    def initialize(self,L=None):
        if L is None:
            self.L = torch.ones(self.batch_size,device=self.device)*0.001
        else:
            self.L = L

    def update(self,function):

        self.function = function
        self.batch_size = function.batch_size
        self.lengths = function.lengths
        self.max_length = self.lengths.max()
        (matrix,lengths) = self.function.linearMatrix()
        self.nplengths = self.lengths.cpu().numpy()
        
        map_inputs = [(matrix[i].cpu().numpy(),self.nplengths[i]) for i in range(self.batch_size)]
        outputs = np.stack(list(map(Eisner,map_inputs)))
        outputs = torch.tensor(outputs,dtype=torch.float,device=self.device)
        self.outputs = outputs


        self.initialize()
        self.xt = [outputs[i,0:self.lengths[i],0:self.lengths[i]].requires_grad_(True) for i in range(self.batch_size)]
        self.padxt = utils.padding_structure(self.xt,self.batch_size,self.device,self.max_length)
        self.valuet = self.function.value(self.xt)
        #print('inital energy of FW: ' + str(self.valuet.detach()))
        #exit() 
        self.gradt = self.function.grad()
        self.padgradt = utils.padding_structure(self.gradt,self.batch_size,self.device,self.max_length)
        if (not(self.version=='AdaFW')):
            self.S = [outputs[i].unsqueeze(0) for i in range(self.batch_size)]
            self.alpha = [torch.tensor([1.0],dtype=torch.float,device=self.device) for i in range(self.batch_size)]
        self.tpadxt = self.padxt
        self.zero_index_gamma = None
        self.restart_index = None
        self.zero_index_gt = None
        self.active = None
        self.allgt = None
        self.tlengths = self.lengths

    def Qt(self,gamma,M,energyt):

        delta = -gamma*self.gt + 0.5*gamma*gamma*M*((self.dt**2).sum(dim=(1,2)))
        EQt = energyt+delta
        return EQt,delta
    
    def step(self):

        map_inputs = [(-self.padgradt[i].cpu().numpy(),self.tlengths[i].item()) for i in range(self.batch_size)]
        outputss = np.stack(list(map(Eisner,map_inputs)))
        self.st = torch.tensor(outputss, dtype=torch.float, device=self.device)
        if (self.version=='AdaFW'):
            self.dt = self.st-self.padxt
            self.gamma_max = torch.ones(self.batch_size,device=self.device)
        else:
            index_s = [torch.all(torch.eq(self.st[i].reshape(-1),self.S[i].reshape(self.S[i].size()[0],-1)),dim=1) for i in range(self.batch_size)]
            values = [(self.padgradt[i]*(self.tpadxt[i].detach()-self.S[i])).reshape(self.S[i].size()[0],-1).sum(1) for i in range(self.batch_size)]
            index_v = [torch.argmin(values[i]).item() for i in range(self.batch_size)]
            self.vt = torch.stack([self.S[i][index_v[i]] for i in range(self.batch_size)])

            if (self.version=='AdaPFW'):
                self.dt = self.st-self.vt
                self.gamma_max = torch.tensor([self.alpha[i][index_v[i]] for i in range(self.batch_size)],dtype=torch.float,device=self.device)
            else:
                dt_fw = self.st - self.padxt
                dt_aw = self.padxt - self.vt
                delta_fw = (self.padgradt*dt_fw).sum(dim=(1,2))
                delta_aw = (self.padgradt*dt_aw).sum(dim=(1,2))
                index_fa = delta_fw <= delta_aw
                self.dt = dt_fw.new_empty(dt_fw.size(),device=self.device)
                self.dt[index_fa] = dt_fw[index_fa]
                self.dt[~index_fa] = dt_aw[~index_fa]
                self.gamma_max = torch.tensor([1 if index_fa[i] else self.alpha[i][index_v[i]]/(1-self.alpha[i][index_v[i]]) for i in range(self.batch_size)],type=torch.float,device=self.device)

        self.gt = -(self.padgradt*self.dt).sum(dim=(1,2))
        if self.allgt is None:
            self.allgt = self.gt
        else:
            self.allgt[self.active] = self.gt
        index_gt = self.gt>=self.tolerance
        if self.active is None:
            self.active = index_gt
        else:
            if self.zero_index_gt is not None:
                index_gt[self.zero_index_gt] = False
            self.active[self.active] = index_gt
        if not(any(self.active)==True):
            return self.padxt.detach(), self.allgt, self.active
        
        #adaptivity process
       
        indices = [i for i,x in enumerate(index_gt) if x]
        self.batch_size = (1*self.active).sum()
        self.L = self.L[index_gt]
        
        self.tlengths = self.tlengths[index_gt]
        self.gt = self.gt[index_gt]
        index_s = [index_s[i] for i in indices]
        index_v = [index_v[i] for i in indices]
        self.dt = self.dt[index_gt]
        self.st = self.st[index_gt]
        self.vt = self.vt[index_gt]
        self.gamma_max = self.gamma_max[index_gt]
        self.tpadxt = self.tpadxt[index_gt]
        energyt = self.valuet[self.active]
        if self.restart_index is not None: 
            self.restart_index = [self.restart_index[i] for i in indices]
        self.S = [self.S[i] for i in indices]
        self.alpha = [self.alpha[i] for i in indices]



        M = self.L/self.eta
        minM = self.gt/(self.gamma_max*((self.dt**2).sum(dim=(1,2))))
        tindex = M<minM
        M[tindex] = minM[tindex]
        gamma_c = self.gt/(M*((self.dt**2).sum(dim=(1,2))))
        gamma = torch.min(gamma_c,self.gamma_max)
        delta = (gamma*self.dt.reshape(self.batch_size,-1).transpose(0,1)).transpose(0,1).reshape(self.batch_size,self.max_length,self.max_length)
        xt = self.tpadxt + delta
        Et = self.function.value(xt,self.active,True)
        index_energy_best = Et<energyt
        energy_best = Et.new_empty(Et.size())
        energy_best[index_energy_best] = Et[index_energy_best]
        energy_best[~index_energy_best] = energyt[~index_energy_best]
        gamma_temporary = gamma.new_zeros(gamma.size())
        gamma_temporary[index_energy_best] = gamma[index_energy_best]
        M_temporary = self.L.new_empty(self.L.size())
        M_temporary[index_energy_best] = M[index_energy_best]
        M_temporary[~index_energy_best] = self.L[~index_energy_best]
        EQt,dEQt = self.Qt(gamma,M,energyt)
        index_e = Et>EQt
        aactive = copy.deepcopy(self.active)
        aactive[self.active] = index_e
        while any(index_e)==True:
            M[index_e] = M[index_e]*self.tau
            gamma_c = self.gt/(M*((self.dt**2).sum(dim=(1,2))))
            gamma = torch.min(gamma_c,self.gamma_max)
            delta = (gamma*self.dt.reshape(self.batch_size,-1).transpose(0,1)).transpose(0,1).reshape(self.batch_size,self.max_length,self.max_length)
            xt = self.tpadxt + delta
            xt = xt[index_e]
            with torch.no_grad(): tEt = self.function.value(xt,aactive,True)
            Et[index_e] = tEt
            #compare if energy better than the previous
            index_energy_best = Et<energy_best
            energy_best[index_energy_best] = Et[index_energy_best]
            gamma_temporary[index_energy_best] = gamma[index_energy_best]
            M_temporary[index_energy_best] = M[index_energy_best]
            EQt,dEQt = self.Qt(gamma,M,energyt)
            index_e = Et>EQt
            aactive[self.active] = index_e
        
        self.zero_index_gamma = (gamma_temporary==0)
        if self.restart_index is not None:
            self.zero_index_gt = [True if self.zero_index_gamma[i] and self.restart_index[i] else False for i in range(self.batch_size)]
        #set restart for points get gamma zero but do not arrive the tolerance
        self.restart_index = [True if self.zero_index_gamma[i] and index_gt[i] else False for i in range(self.batch_size)]
        self.L = M_temporary
        self.gammat = gamma_temporary
        delta = (self.gammat*self.dt.reshape(self.batch_size,-1).transpose(0,1)).transpose(0,1).reshape(self.batch_size,self.max_length,self.max_length)
        oldtpadxt = copy.deepcopy(self.tpadxt.detach())
        self.tpadxt = self.tpadxt.detach() + delta
        
        #add the noise of lagevin dynamic
        noise = torch.randn(self.tpadxt.size()).numpy()
        map_inputs = [(noise[i], self.nplengths[i]) for i in range(self.batch_size)]
        outputs = np.stack(list(map(Eisner, map_inputs)))
        self.noise = torch.tensor(outputs, dtype=torch.float, device=self.device)
        index_n = [torch.all(torch.eq(self.noise[i].reshape(-1),self.S[i].reshape(self.S[i].size()[0],-1)),dim=1) for i in range(self.batch_size)]
        self.ndt = self.noise - self.tpadxt
        self.ngammat = torch.min(torch.sqrt(2*self.gammat),self.gammat.new_ones(self.gammat.size()))
        ndelta = (self.ngammat*self.dt.reshape(self.batch_size,-1).transpose(0,1)).transpose(0,1).reshape(self.batch_size,self.max_length,self.max_length)
        self.tpadxt = self.tpadxt.detach() + ndelta
        
        self.padxt[self.active] = self.tpadxt
        
        
        xt = [self.tpadxt[i,0:self.tlengths[i],0:self.tlengths[i]].detach().requires_grad_(True) for i in range(self.batch_size)]
        txt = utils.padding_structure(xt,self.batch_size,self.device,self.max_length).detach()
        indexs = txt==self.tpadxt.detach()

        valuet = self.function.value(xt,self.active)
        gradt = self.function.grad()
        index_valuet = valuet!=valuet
        for i in range(self.batch_size):
            if torch.any(gradt[i]!=gradt[i]):
                print('first test')
                print(xt[i])
                print(valuet[i])
                print(gradt[i])
                exit()
        
        self.padgradt = utils.padding_structure(gradt,len(gradt),self.device,self.max_length)
        index_valuet = valuet!=valuet
        if any(index_valuet):
            print('first test')
            print(self.padgradt[index_valuet])
            print(txt[index_valuet])
            print(self.dt[index_valuet])
            exit()

        self.valuet[self.active] = valuet
        if not(self.version=='AdaFW'):
            #update S and alpha
            if self.version=='AdaPFW':
                for i in range(self.batch_size):
                    if self.restart_index[i]:
                        #self.S[i] = self.tpadxt[i].detach().unsqueeze(0)
                        #self.alpha[i] = torch.tensor([1.0],dtype=torch.float,device=self.device)
                        self.L[i] = 0.001
                    else:
                        self.alpha[i][index_v[i]] -= self.gammat[i]
                        if any(index_s[i])==False:
                            self.alpha[i]=torch.cat((self.alpha[i],self.gammat[i].unsqueeze(0)))
                            self.S[i] = torch.cat((self.S[i],self.st[i].unsqueeze(0)))
                        else:
                            self.alpha[i][index_s[i]] += self.gammat[i]
                index_n = [torch.all(torch.eq(self.noise[i].reshape(-1),self.S[i].reshape(self.S[i].size()[0],-1)),dim=1) for i in range(self.batch_size)]

                #treatement of noise
                for i in range(self.batch_size):
                    if not(self.restart_index[i]):
                        self.alpha[i] = self.alpha[i]*(1-self.ngammat[i])
                        if any(index_n[i])==False:
                            self.alpha[i] = torch.cat((self.alpha[i], self.ngammat[i].unsqueeze(0)))
                            self.S[i] = torch.cat((self.S[i], self.noise[i].unsqueeze(0)))
                        else:
                            self.alpha[i][index_n[i]] += self.ngammat[i]
                        
            else:
                for i in range(self.batch_size):
                    if index_fa[i]:
                        self.alpha[i] = self.alpha[i]*(1-self.gammat[i])
                        if index_s[i].nelement()==0:
                            self.alpha[i]=torch.cat((self.alpha[i],self.gammat[i].unsqueeze(0)))
                            self.S[i]== torch.cat((self.S[i],self.st[i].unsqueeze(0)))
                        else:
                            self.alpha[i][index_s[i][0]] += self.gammat[i]
                    else:
                        self.alpha[i] = self.alpha[i]*(1+self.gammat[i])
                        self.alpha[i][index_v[i]] -= self.gammat[i]
            #delete zero S and alphas
            index_non_zero = [~(self.alpha[i]==0) for i in range(self.batch_size)]
            for i in range(self.batch_size):
                if all(index_non_zero[i])==False:
                    self.alpha[i]=self.alpha[i][index_non_zero[i]]
                    self.S[i]=self.S[i][index_non_zero[i]]
        #print('energy')
        #print(self.valuet)
        index_energy_nan = self.valuet!=self.valuet
        if any(index_energy_nan):

            print(self.padxt[index_energy_nan])
            
            exit()
        #exit()
        return self.padxt.detach(), self.allgt, self.active
    



