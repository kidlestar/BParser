import torch
import numpy as np
from topk import parse_proj

def Eisner(input_data):
    (energy,length,tk) = input_data
    gmatrix = energy[0:length,0:length]
    ngmatrix = np.zeros((length+1,length+1))
    predict = np.zeros((tk,energy.shape[0],energy.shape[1]))
    idx = np.arange(length)
    ngmatrix[0,idx+1] = gmatrix[idx,idx]
    ngmatrix[1:,1:] = gmatrix
    head = parse_proj(ngmatrix,tk)
    for i in range(length):
        for j in range(tk):
            if head[i+1,j]==0: predict[j,i,i] = 1
            else: predict[j,head[i+1,j]-1,i] = 1
    return predict



class EisnerK:

    def __init__(self, k=10):
        self.k = k

    def update(self, function):
        self.function = function
        self.device = function.net.device
        self.batch_size = function.batch_size
        self.lengths = function.lengths.cpu().numpy()
        #print('lengths of sentences: ' + str(self.lengths))
        self.le = self.function.net.linearMatrix.detach().cpu().numpy()
        self.mask = torch.tensor([True]*self.batch_size, device=self.device)

    def sample(self):
        data_input = [(-self.le[i], self.lengths[i], self.k) for i in range(self.batch_size)]
        data_output = np.array(list(map(Eisner, data_input))).transpose((1,0,2,3))
        data = torch.tensor(data_output, dtype=torch.float, device=self.device)
        return data
    
    def bestk(self):
        #construct the input data
        data_input = [(-self.le[i], self.lengths[i], self.k) for i in range(self.batch_size)]
        data_output = np.array(list(map(Eisner, data_input))).transpose((1,0,2,3))
        data = torch.tensor(data_output, dtype=torch.float, device=self.device)
        eng = torch.empty(self.k,self.batch_size, device=self.device)
        with torch.no_grad():
            for i in range(self.k):
                if self.function.net.training:
                    le, ge = self.function.net.NetEnergy(data[i], True, self.mask)
                    eng[i] = le + ge
                else:
                    eng[i] = self.function.net.NetEnergy(data[i], True, self.mask)
        #print(eng)
        #get the smallest value of every column
        me, mindices = eng.min(0)
        print(mindices)
        mins = torch.empty(self.batch_size, data.size()[2], data.size()[3], device=self.device)
        idx = torch.arange(0,self.batch_size,device=self.device).long()
        
        mins[idx] = data[mindices,idx]

        return mins



