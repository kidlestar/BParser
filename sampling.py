import torch
from torch_struct import DependencyCRF


#sampling by Metropolis Hastling Algorithm
class MHSampling:

    def __init__(self, k=10):
        self.k = k

    def update(self, function):
        self.function = function
        self.device = self.function.net.device
        self.batch_size = function.batch_size
        self.lengths = function.lengths
        self.max_length = self.lengths.max()
        self.le = self.function.net.linearMatrix.detach()
        self.mask = torch.tensor([True]*self.batch_size, device=self.device)
        self.deptree = DependencyCRF(-self.le, self.lengths)
        self.x = self.deptree.argmax.detach()
        self.p = torch.exp(self.deptree.log_prob(self.x)).detach()
        #with torch.no_grad():
        el, eg = function.net.NetEnergy(self.x, False, self.mask)
        self.ret = torch.empty(self.k+1, self.batch_size, device=self.device)
        self.ret[0] = el + eg
        self.e = self.ret[0].detach()

    def sample(self):
        for i in range(self.k):
            self.nx = self.deptree.sample([1]).squeeze(0).detach()
            self.np = torch.exp(self.deptree.log_prob(self.nx)).detach()
            with torch.no_grad():
                nel, neg = self.function.net.NetEnergy(self.nx, True, self.mask)
                self.ne = nel + neg
            self.A = torch.min(self.ne.new_ones(self.ne.size()),torch.exp(self.e-self.ne)*self.p/self.np)
            u = torch.rand(self.A.size(),device=self.device)
            index = u<=self.A
            self.tx = torch.zeros(self.x.size(), device=self.device)
            self.tx[index] = self.nx[index]
            self.tx[~index] = self.x[~index]
            self.p[index] = self.np[index]
            self.e[index] = self.ne[index]
            tel, teg = self.function.net.NetEnergy(self.tx, False, self.mask)
            self.ret[i+1] = tel + teg
        return self.ret.mean(0)



#Importance Sampling according to the know linear probability
class IMSampling:
    
    def __init__(self, k=10):
        self.k = k
        self.softmax = torch.nn.Softmax(dim=0)

    def update(self, function):
        self.function = function
        self.device = self.function.net.device
        self.batch_size = function.batch_size
        self.lengths = function.lengths
        self.max_length = self.lengths.max()
        self.le = self.function.net.linearMatrix.detach()
        self.mask = torch.tensor([True]*self.batch_size, device=self.device)
        self.deptree = DependencyCRF(-self.le, self.lengths)
        self.ke = torch.zeros(self.k, self.batch_size, device=self.device)

    def weight(self, ke, kp):
        e = -ke-kp
        w = self.softmax(e)
        return w

    def sample(self):
        #sample k trees according to the linear part
        self.kx = self.deptree.sample([self.k]).detach()
        self.klp = self.deptree.log_prob(self.kx).detach()
        for i in range(self.k):
            el, eg = self.function.net.NetEnergy(self.kx[i], False, self.mask)
            self.ke[i,:] = el + eg
        self.w = self.weight(self.ke.detach(), self.klp)
        print(self.w[:,0])
        self.e = (self.w * self.ke).sum(0)
        return self.e


