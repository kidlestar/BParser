import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import time
def reverse(lst: List[Tensor])->List[Tensor]:
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]


class CusRNNCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(CusRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.whh = torch.empty(self.weight_hh.size(), device = self.weight_hh.device)
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(hidden_size))

    @jit.script_method
    def forward(self, iinput, state, is_first=False):
        hx = state
        if is_first:
          self.whh = self.weight_hh/torch.sqrt((self.weight_hh**2).sum(0))/10.0
        whh = self.whh
        bhh = self.bias_hh
        wih = self.weight_ih
        bih = self.bias_ih

        p1 = torch.relu(torch.matmul(iinput,torch.abs(wih.t()))+bih)
        p2 = torch.relu(torch.matmul(hx,torch.abs(whh))+bhh)
        output=p1+p2
        
        return output, output

    @jit.script_method
    def evaluate(self, iinput, state, is_first=False):
        hx = state
        if is_first:
          self.whh = self.weight_hh/torch.sqrt((self.weight_hh**2).sum(0))/10.0
        whh = self.whh.detach()
        bhh = self.bias_hh.detach()
        wih = self.weight_ih.detach()
        bih = self.bias_ih.detach()

        p1 = torch.relu(torch.matmul(iinput,torch.abs(wih.t()))+bih)
        p2 = torch.relu(torch.matmul(hx,torch.abs(whh))+bhh)
        output=p1+p2
        return output, output


class CusRNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(CusRNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, iinput, state):
        inputs = iinput.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs,dim=1), state

    @jit.script_method
    def evaluate(self, iinput, state):
        inputs = iinput.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell.evaluate(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs,dim=1), state

class ReverseCusRNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseCusRNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, iinput, state):
        #inputs = reverse(iinput.unbind(1))
        inputs = torch.flip(iinput,(1,))
        #inputs = iinput
        inputs = inputs.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            if i==0: 
                out, state = self.cell(inputs[i], state, True)
            else:
                out, state = self.cell(inputs[i], state)
            outputs += [out]
        outputs = torch.stack(outputs,dim=1)
        outputs = torch.flip(outputs,(1,))
        #return torch.stack(reverse(outputs),dim=1), state
        return outputs, state

    @jit.script_method
    def evaluate(self, iinput, state):
        #inputs = reverse(iinput.unbind(1))
        inputs = torch.flip(iinput,(1,))
        #inputs = iinput
        inputs = inputs.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            if i==0:
                out, state = self.cell.evaluate(inputs[i], state, True)
            else:
                out, state = self.cell.evaluate(inputs[i], state)
            outputs += [out]
        outputs = torch.stack(outputs,dim=1)
        outputs = torch.flip(outputs,(1,))
        return outputs, state
        #return torch.stack(reverse(outputs),dim=1), state

class BiCusRNNLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BiCusRNNLayer, self).__init__()
        self.directions = nn.ModuleList([
            CusRNNLayer(cell, *cell_args),
            ReverseCusRNNLayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, iinput, state):
        outputs = jit.annotate(List[Tensor], [])
        for direction in self.directions:
            out, out_state = direction(iinput, state)
            outputs += [out]
        return torch.cat(outputs, 2)

    @jit.script_method
    def evaluate(self, iinput, state):
        outputs = jit.annotate(List[Tensor], [])
        for direction in self.directions:
            out, out_state = direction.evaluate(iinput, state)
            outputs += [out]
        return torch.cat(outputs, 2)

