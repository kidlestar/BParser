import torch
import torch.nn as nn
from torch.nn import Parameter
#import torch.jit as jit
from torch import Tensor
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence
import numpy as np
import math
import time

from modules.dropout import SharedDropout_convex


class CusRNNCell(RNNCellBase):
    #__constants__ = ['input_size','hidden_size']
    def __init__(self, input_size, hidden_size, bias=True):
        super(CusRNNCell, self).__init__(input_size,hidden_size,bias,1)
        self.mask = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        #self.dropout = dropout
        #self.device = device
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        #self.bias_ih = Parameter(torch.randn(hidden_size))
        #self.bias_hh = Parameter(torch.randn(hidden_size))
        self.reset_parameters()
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def reset_parameters(self):
        """
        for i in self.parameters():
            if len(i.shape) > 1:
                #nn.init.orthogonal_(i)
                #nn.init.kaiming_normal_(i,a=0.1,mode='fan_out',nonlinearity='leaky_relu')
            else:
                nn.init.zeros_(i)
        """
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        #nn.init.kaiming_normal_(self.weight_ih,a=0.1,mode='fan_in',nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.weight_hh,a=0.1,mode='fan_out',nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, input, state, is_flag=False):
        hx = state 
        if is_flag:
            self.whh = self.weight_hh/torch.sqrt((self.weight_hh*self.weight_hh).sum(0))/np.sqrt(self.hidden_size)
        whh = self.whh
        bhh = self.bias_hh
        wih = self.weight_ih
        bih = self.bias_ih

        p1 = self.activation(torch.matmul(input,torch.abs(wih.t()))+bih)
        p2 = self.activation(torch.matmul(hx,torch.abs(whh))+bhh)
        output=p1+p2
        """
        if self.training:
            if self.mask is None:
                self.mask = SharedDropout_convex.get_mask(output,self.dropout)
                self.fix_mask = self.mask
            output = output * self.mask
        """
        #res = self.dropout(output)
        return output

    def evaluate(self, input, state, is_flag=False):
        hx = state
        if is_flag:
            self.whh = self.weight_hh/torch.sqrt((self.weight_hh*self.weight_hh).sum(0))/np.sqrt(self.hidden_size)
        whh = self.whh.detach()
        bhh = self.bias_hh.detach()
        wih = self.weight_ih.detach()
        bih = self.bias_ih.detach()
        p1 = self.activation(torch.matmul(input,torch.abs(wih.t()))+bih)
        p2 = self.activation(torch.matmul(hx,torch.abs(whh))+bhh)
        output=p1+p2
        """
        if self.training:
            if self.mask is None:
                self.mask = SharedDropout_convex.get_mask(output,self.dropout)
                self.fix_mask = self.mask
            output = output * self.mask
        """
        #res = self.dropout(output)
        return output



class CusRNNLayer(nn.Module):
    def __init__(self, cell, dropout, *cell_args):
        super(CusRNNLayer, self).__init__()
        self.dropout = dropout
        self.cell = cell(*cell_args)
        self.hid_mask = None
        self.hid_mask_fix = None

    def forward(self, input, state, batch_sizes, reverse=False):
        init_state = state
        outputs, seq_len = [],len(input)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training and self.dropout!=0:
            if self.hid_mask is None: 
                self.hid_mask = SharedDropout_convex.get_mask(state,self.dropout)
                self.hid_mask_fix = self.hid_mask
            hid_mask = self.hid_mask
        i=0
        for t in steps:
            last_batch_size, batch_size = len(state), batch_sizes[t]
            if last_batch_size<batch_size:
                state = torch.cat((state,init_state[last_batch_size:batch_size]))
            else:
                state = state[:batch_size]
            if i==0:
                state = self.cell(input[t], state, True)
            else:
                state = self.cell(input[t], state)
            if self.training and self.dropout!=0: state = state * hid_mask[:batch_size]
            outputs += [state]
            i+=1
        if reverse: outputs.reverse()
        return torch.cat(outputs), state

    def evaluat(self, input, state, batch_sizes, reverse=False):
        init_state = state
        outputs, seq_len = [],len(input)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training and self.dropout!=0:
            if self.hid_mask is None: 
                self.hid_mask = SharedDropout_convex.get_mask(state,self.dropout)
                self.hid_mask_fix = self.hid_mask
            hid_mask = self.hid_mask
        i=0
        for t in steps:
            last_batch_size, batch_size = len(state), batch_sizes[t]
            if last_batch_size<batch_size:
                state = torch.cat((state,init_state[last_batch_size:batch_size]))
            else:
                state = state[:batch_size]
            if i==0:
                state = self.cell.evaluate(input[t], state, True)
            else:
                state = self.cell.evaluate(input[t], state)
            if self.training and self.dropout!=0: state = state * hid_mask[:batch_size]
            outputs += [state]
            i+=1
        if reverse: outputs.reverse()
        return torch.cat(outputs), state
    
    def set(self,mask=None):
        if mask is not None:
            self.hid_mask = self.hid_mask_fix[mask]
            
        else:
            self.hid_mask = None
            self.hid_mask_fix = None


class BiCusRNNLayer(nn.Module):
    def __init__(self, device, CusRNN, cell, dropout, *cell_args):
        super(BiCusRNNLayer, self).__init__()
        self.device = device
        self.dropout = dropout
        self.hidden_size = cell_args[1]
        #self.dropout = SharedDropout_convex(dropout)
        self.RNN1 = CusRNN(cell,dropout,*cell_args)
        self.RNN2 = CusRNN(cell,dropout,*cell_args)
        self.mask=None
        self.mask_fix = None

    def forward(self, input, state=None):
        #rinput = torch.zeros(input.size(),device=self.device)
        #for i in range(lengths.shape[0]):
            #rinput[i,0:lengths[i]+1] = torch.flip(input[i,0:lengths[i]+1], (0,))
        input,batch_sizes,sorted_indice,unsorted_indice = input
        batch_size = batch_sizes[0]
        if state is None: state = input.new_zeros(batch_size,self.hidden_size)
        if self.training and self.dropout!=0:
            if self.mask is None: 
                self.mask = SharedDropout_convex.get_mask(input[:batch_size],self.dropout)
                self.mask_fix = None
            mask = torch.cat([self.mask[:batch_size] for batch_size in batch_sizes])
            input*=mask
        input = torch.split(input,batch_sizes.tolist())
        #input = self.dropout(input)
        outputs, sn = self.RNN1(input,state,batch_sizes,False)
        routputs,rsn = self.RNN2(input,state,batch_sizes,True)
        x=torch.cat((outputs,routputs),-1)
        return PackedSequence(x,batch_sizes,None,None)

    def evaluate(self, input, state=None):
        #rinput = torch.zeros(input.size(),device=self.device)
        #for i in range(lengths.shape[0]):
            #rinput[i,0:lengths[i]+1] = torch.flip(input[i,0:lengths[i]+1], (0,))
        input,batch_sizes,sorted_indice,unsorted_indice = input
        batch_size = batch_sizes[0]
        if state is None: state = input.new_zeros(batch_size,self.hidden_size)
        if self.training and self.dropout!=0:
            if self.mask is None: 
                self.mask = SharedDropout_convex.get_mask(input[:batch_size],self.dropout)
                self.mask_fix = self.mask
            mask = torch.cat([self.mask[:batch_size] for batch_size in batch_sizes])
            input*=mask
        input = torch.split(input,batch_sizes.tolist())
        outputs, sn = self.RNN1.evaluat(input,state,batch_sizes,False)
        routputs,rsn = self.RNN2.evaluat(input,state,batch_sizes,True)
        x=torch.cat((outputs,routputs),-1)
        return PackedSequence(x,batch_sizes,None,None)

    def set(self,mask=None):
        if mask is None:
            self.mask = None
            self.mask_fix = None
        else:
            self.mask = self.mask_fix[mask]

        self.RNN1.set(mask)
        self.RNN2.set(mask)
