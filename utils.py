import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import copy

import calgorithm

def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def Eisner(input_data):
    (energy,length) = input_data
    gmatrix = energy[0:length,0:length]
    ngmatrix = np.zeros((length+1,length+1))
    predict = np.zeros(energy.shape)
    idx = np.arange(length)
    ngmatrix[0,idx+1] = gmatrix[idx,idx]
    ngmatrix[1:,1:] = gmatrix
    ngmatrix[:,0]-=np.inf
    head = calgorithm.parse_proj(ngmatrix)
    #head = calgorithm.parse_proj(ngmatrix)
    for i in range(length):
        if head[i+1]==0: predict[i,i] = 1
        else: predict[head[i+1]-1,i] = 1
    return predict

#project a vector into the unit simplex
def proj(y):
 
    l = y
    idx = np.argsort(l)
    d = len(l)
 
    evalpL = lambda k: np.sum((y[idx[k:]] - l[idx[k]])) -1
 
    
    def bisectsearch():
        idxL, idxH = 0, d-1
        L = evalpL(idxL)
        H = evalpL(idxH)
 
        if L<0:
            return idxL
 
        while (idxH-idxL)>1:
            iMid = int((idxL+idxH)/2)
            M = evalpL(iMid)
 
            if M>0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M
 
        return idxH
 
    k = bisectsearch()
    lam = (np.sum(y[idx[k:]])-1)/(d-k)
 
    x = np.maximum(0, y-lam)
 
    return x

#padding for the input batch of sequence (sent, postag and structure)
def padding_train(sent,char,pos,deprel,structure,batch_size,w2i,i2w,xpos):
    #tsent = copy.deepcopy(sent)
    #tpos = copy.deepcopy(pos)
    #do things similar to data augmentation
    #pad things
    pad_sent = pad_sequence(sent,batch_first=True)
    #pad_bsent = pad_sequence(bsent,batch_first=True)
    pad_pos = pad_sequence(pos,batch_first=True)
    pad_char = pad_sequence(char,batch_first=True)
    pad_deprel = pad_sequence(deprel,batch_first=True)
    lengths = torch.tensor(np.array([_.size()[0] for _ in structure])).long()
    max_length = lengths.max()
    pad_structure = torch.zeros(batch_size,max_length,max_length)
    idx = torch.zeros(batch_size,max_length,max_length)
    for i in range(batch_size):
        pad_structure[i,0:lengths[i],0:lengths[i]] = structure[i]
        idx[i,:lengths[i],:lengths[i]] = 1
    mask = (idx==1)
    mask1 = (pad_deprel!=0)
    mask2 = (pad_sent!=0)
    return pad_sent,pad_char,pad_pos,pad_deprel,pad_structure,lengths, mask, mask1, mask2

#padding for the input batch of sequence (sent, postag and structure)
def padding_evaluate(sent,char,pos,deprel,structure,punct):
    pad_sent = pad_sequence(sent,batch_first=True)
    #pad_bsent = pad_sequence(bsent,batch_first=True)
    pad_char = pad_sequence(char,batch_first=True)
    pad_pos = pad_sequence(pos,batch_first=True)
    pad_deprel = pad_sequence(deprel,batch_first=True)
    pad_punct = pad_sequence(punct,batch_first=True)
    lengths = torch.tensor(np.array([_.size()[0] for _ in structure])).long()
    max_length = lengths.max()
    batch_size = lengths.size()[0]
    delta_lengths = max_length-lengths
    pad_structure = torch.zeros(batch_size,max_length,max_length)
    idx = torch.zeros(batch_size, max_length, max_length)
    for i in range(batch_size):
        pad_structure[i,0:lengths[i],0:lengths[i]] = structure[i]
        idx[i,:lengths[i],:lengths[i]] = 1
    mask = (idx==1)
    mask1 = (pad_deprel!=0)
    mask2 = (pad_sent!=0)
    return pad_sent, pad_char, pad_pos, pad_deprel, pad_structure, pad_punct, lengths, mask, mask1, mask2

def padding_structure(structure,batch_size,device,max_length=None):
    lengths = torch.tensor([_.size()[0] for _ in structure],device=device)
    if max_length is None: max_length = lengths.max()
    
    delta_lengths = max_length - lengths
    pad_structure = torch.zeros(batch_size,max_length,max_length,device=device)
    for i in range(batch_size):
        pad_structure[i,0:lengths[i],0:lengths[i]] = structure[i]
    return pad_structure

def padding_matrix(matrix, max_length):
    batch_size = len(matrix)
    ret = np.zeros((batch_size,max_length,max_length))
    for i in range(batch_size):
        ret[i,0:matrix[i].shape[0],0:matrix[i].shape[1]] = matrix[i]
    return ret

def list_creation(old_energy,new_energy,old_gradient,new_gradient,old_y,new_y,old_xY,new_xY,is_restart,theta,learning_rate,batch_size):
    result = []
    for i in range(batch_size):
        result.append((old_energy[i],new_energy[i],old_gradient[i],new_gradient[i],old_y[i],new_y[i],old_xY[i],new_xY[i],is_restart[i],theta[i],learning_rate[i]))
    return result

def initialize_structure(length,device):
    ret = np.zeros((length,length))
    ret[0,:] = 1.0
    return torch.tensor(ret,dtype=torch.float,device=device,requires_grad=True)

def initialize_set(length):
    ret = np.zeros((length,length))
    ret[0,:] = 1.0
    return [ret]

def set_zero(devisior):
    devisior[devisior==0] = 1.0
    return devisior

def energy(input):
    return input.item()

get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if np.array_equal(x,y)]
get_zero_indexes = lambda xs: [i for (y,i) in zip(xs, range(len(xs))) if y==0]


#evaluate the training data
def evaluate(device,data,model,fn,old,ITER,ideprel):
    #res = []
    correct = 0.0
    rcorrect = 0.0
    correct_ = 0.0
    rcorrect_ = 0.0
    kcorrect = 0.0
    cword = 0.0
    batch_size = 128
    nc = 0.0
    if len(data.data)%batch_size==0: num_batch = int(len(data.data)/batch_size)
    else: num_batch = int(len(data.data)/batch_size)+1
    #for i_batch, sample_batched in enumerate(devloader):
    for i in range(num_batch):
        
        if i==(num_batch-1):
            sent = data.torchSent[i*batch_size:]
            #bsent = data.torchBSent[i*batch_size:]
            char = data.torchChar[i*batch_size:]
            pos = data.torchPostag[i*batch_size:]
            structure = data.torchArbores[i*batch_size:]
            deprel = data.torchDepreltag[i*batch_size:]
            punct = data.torchPunct[i*batch_size:]
            orisents = data.data[i*batch_size:]
        else:
            sent = data.torchSent[i*batch_size:(i+1)*batch_size]
            #bsent = data.torchBSent[i*batch_size:(i+1)*batch_size]
            char = data.torchChar[i*batch_size:(i+1)*batch_size]
            pos = data.torchPostag[i*batch_size:(i+1)*batch_size]
            deprel = data.torchDepreltag[i*batch_size:(i+1)*batch_size]
            structure = data.torchArbores[i*batch_size:(i+1)*batch_size]
            punct = data.torchPunct[i*batch_size:(i+1)*batch_size]
            orisents = data.data[i*batch_size:(i+1)*batch_size]
        pad_sent,pad_char, pad_pos,pad_deprel, pad_structure, pad_punct, lengths, mask, mask1, mask2 = padding_evaluate(sent, char, pos, deprel, structure, punct)
        
        pad_sent = pad_sent.to(device)
        #pad_bsent = pad_bsent.to(device)
        pad_char = pad_char.to(device)
        pad_pos = pad_pos.to(device)
        pad_structure = pad_structure.to(device)
        pad_deprel = pad_deprel.to(device)
        pad_punct = pad_punct.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        Tbatch_size = lengths.size()[0]
        
        pad_bsent = None
        
        #Ppredict, kPpredict = model.evaluate(pad_sent,pad_pos,Tbatch_size,lengths,max_iter=min(30+100*ITER,3000))
        Ppredict, Ppredict_, Rpredict = model.evaluate(pad_sent, pad_bsent, pad_char, pad_pos,Tbatch_size,lengths,mask, mask1, mask2)

        rel = Rpredict.transpose(-1,-2)[Ppredict.transpose(-1,-2)==1]
        rel_ = Rpredict.transpose(-1,-2)[Ppredict_.transpose(-1,-2)==1]
        trel = pad_deprel[mask1]
        tmask = (Ppredict*pad_structure).sum(dim=-2)>0
        tmask_ = (Ppredict_*pad_structure).sum(dim=-2)>0
        
        mpunc = (pad_punct==1)
        
        tmask[mpunc] = False
        tmask_[mpunc] = False
        
        tmask = tmask[mask1]
        tmask_ = tmask_[mask1]
        rcorrect += ((rel==trel) & tmask).sum().item()
        rcorrect_ += ((rel_==trel) & tmask_).sum().item()

        predict_trees = Ppredict.cpu().numpy()
        #map_inputs = [(kPpredict[i],lengths[i].item()) for i in range(Tbatch_size)]
        #kpredict_trees = np.array(list(map(Eisner,map_inputs)))
        

        #kpredict_trees = kPpredict
        
        #predict_trees = Ppredict
        #outputs = DepTree(MaxSemiring).marginals(Ppredict, lengths=lengths)
        #change the head of data
        #calculate the UAS
        #kcorrect += (kpredict_trees*pad_structure.numpy()).sum()
        #correct += (predict_trees*pad_structure.cpu().numpy()).sum()
        correct += tmask.sum().item()
        correct_ += tmask_.sum().item()
        cword += lengths.sum().item() - pad_punct.sum().item()
        """   
        n = 0
        for j in range(Tbatch_size):
            base = i*batch_size
            #temp = Ppredict[j,0:lengths[j],0:lengths[j]].detach().cpu().numpy()
            #np.save('matrix_to_parser.npy',temp)
            #head, temp = Eisner((temp,lengths[j]))
            temp = predict_trees[j,0:lengths[j],0:lengths[j]].transpose()
            #print(head)
            #print(temp)
            #np.save('strange_head_result.npy',head)
            (midx,hidx) = np.where(temp==1)
            for k in range(lengths[j]):
                if midx[k]==hidx[k]: 
                    data.data[base+j][k].head = str(0)
                else: 
                    data.data[base+j][k].head = str(hidx[k]+1)
                data.data[base+j][k].deprel = ideprel[rel[n].item()+1]
                n+=1
        """
    UAS = correct/cword
    LAS = rcorrect/cword
    UAS_ = correct_/cword
    LAS_ = rcorrect_/cword
    #kUAS = kcorrect/cword
    #if model.net.MODE!='Linear': write_conll(fn,data.data)
    #else:
    #if UAS>old: write_conll(fn,data.data)
    #write_conll(fn,data.data)
    return UAS, LAS, UAS_, LAS_
    
    #return 1-nc/cword/2.0

def write_conll(fn, data):
    with open(fn, 'w') as fh:
        for sentence in data:
            for token in sentence:
                fh.write(token.conll() + '\n')
            fh.write('\n')

    """
        for j in range(Tbatch_size):
            predict = Ppredict[j,0:lengths[j],0:lengths[j]]
            temp = structure[j].numpy() * predict.numpy()
            if not(len(orisents[j]) == temp.shape[1]):
                print('wtf')
                exit()
            if np.trace(temp) == 1:
                UAS += 1
            popindex = []
            for k in range(len(orisents[j])):
                if orisents[j][k].form in string.punctuation:
                    popindex.append(k)
            newtemp = np.delete(temp,popindex,1)
            numC = np.sum(newtemp)
            WLAS += numC
            cword += newtemp.shape[1]
    UAS /= data.data_len
    WLAS /= cword
    print('UAS: '+ str(UAS),flush=True)
    print('WLAS: ' + str(WLAS),flush=True)
    """
