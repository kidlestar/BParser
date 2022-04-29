import torch
import torch.nn as nn
import pyconll
import numpy as np
import time
import random
from collections import defaultdict, Counter
from itertools import count
import pickle
import re
import unicodedata

import utils


#This file is for clean data and save into clear files for pytorch

#for setting numbers to NUM
numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    #return 'NUM' if numberRegex.match(word) else word.lower()
    return word.lower()

def ispunct(token):
    if all(unicodedata.category(char).startswith('P') for char in token): return 1
    else: return 0

def isprojective(sequence):
    arcs = [(h, d) for d, h in enumerate(sequence[1:], 1) if h >= 0]
    for i, (hi, di) in enumerate(arcs):
        for hj, dj in arcs[i+1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if (li <= hj <= ri and hi == dj) or (lj <= hi <= rj and hj == di):
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True

class Preprocessing:
    #initailization
    def __init__(self, data_address, glove=None, isTrain=True, w2i=None, c2i=None, xpos=None, deprel=None, chunks=None, buckets=None):
        self.odata = pyconll.load_from_file(data_address)
        print(len(self.odata))
        self.fix_len = 20
        self.isTrain = isTrain
        self.glove = glove
        self.dataClean()
        #if isTrain: self.dataClean()
        self.data_len = len(self.data)
        self.word_len = 0
        self.frequency()
        self.w2i = self.word2index(self.data)
        self.VOCAB_SIZE = len(self.w2i)
        if not(w2i==None): self.w2i = w2i
        if self.isTrain:
            if self.glove is not None: self.embedding()
        
        if not(c2i==None): self.c2i = c2i
        else:self.c2i = self.char2index()
        if not(xpos==None): self.xpos = xpos
        else:self.xpos = self.xposset()
        if not(deprel==None): self.deprel = deprel
        else:self.deprel = self.deprelset()
        self.ideprel = self.ideprelset(self.deprel)
        self.i2w = self.index2word(self.w2i)
        self.createTrainingCorpus()
        self.CHAR_SIZE = len(self.c2i)
        self.torchConversion()
        if chunks is None or buckets is None:
            self.chunk()
        else:
            self.chunks = chunks
            self.buckets = buckets
        #self.save()

    #function for deleting sentences with only one token(not necessary for training or validation)
    def dataClean(self):
        tempData = []
        for sent in self.odata:
            nsent = [token for token in sent if token.head is not None]
            #for token in nsent:
            #    if token.form is None: token.form = '_'
            tempData.append(nsent)
            #flag = isprojective([0]+sequence)
            #if flag: tempData.append(sent)
        self.data=tempData
        print(len(self.data))
    
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
        #return sibs[1:]

    #create the set for char
    def char2index(self):
        c2i = {}
        i = 3
        c2i['<pad>'] = 0
        c2i['<unk>'] = 1
        c2i['<bos>'] = 2
        """
        for sent in self.data:
            for token in sent:
                for c in normalize(token.form):
                    if not(c in c2i.keys()):
                        c2i[c] = i
                        i+=1
        """
        for token, freq in self.ccounter.items():
            if freq >= 1: 
                c2i[token] = i
                i+=1
        
        return c2i
    
    #create the set for pos
    def xposset(self):
        xpos = {}
        i=2
        xpos['*PAD*'] = 0
        xpos['*ROOT-POS*'] = 1
        for sent in self.data:
            for token in sent:
                if not(token.upos == None):
                    if not(token.upos.upper() in xpos.keys()):
                        xpos[token.upos.upper()] = i
                        i+=1
        xpos['*UNK*'] = i
        return xpos
   
    def add_deprel(self, dev, test):
        print(self.deprel)
        print(self.ideprel)
        dev = pyconll.load_from_file(dev)
        test = pyconll.load_from_file(test)
        dep = set()
        for sent in dev:
            for token in sent:
                if token.head is not None: dep.add(token.deprel)
        for sent in test:
            for token in sent:
                if token.head is not None: dep.add(token.deprel)
        #add to self.deprel
        index = len(self.deprel) + 1
        for ndep in list(dep):
            if ndep not in self.deprel.keys():
                self.deprel[ndep] = index
                index += 1
        self.ideprel = self.ideprelset(self.deprel)
        print(self.deprel)
        print(self.ideprel)

    #create the set for deprel
    def deprelset(self):
        deprel = {}
        i=2
        deprel['*UNK*'] = 1
        for sent in self.data:
            for token in sent:
                if not(token.deprel == None):
                    if not(token.deprel in deprel.keys()):
                        deprel[token.deprel] = i
                        i+=1
        return deprel

    #create the set for deprel
    def ideprelset(self, deprel):
        ideprel = {}
        for key in deprel.keys():
            ideprel[deprel[key]] = key.lower()
        return ideprel

    #word to index
    def word2index(self,data):
        w2i = {}
        i = 3
        w2i['<pad>']=0
        w2i['<unk>']=1
        w2i['<bos>']=2
        """
        for sent in data:
            for token in sent:
                if not(normalize(token.form) in w2i.keys()) and self.wordNum[normalize(token.form)]>=2:
                    w2i[normalize(token.form)] = i
                    i += 1
        """
        for token, freq in self.wcounter.items():
            if freq >= 2:
                w2i[token] = i
                i+=1
        return w2i
    
    #index to word
    def index2word(self,w2i):
        i2w = {}
        for key in w2i.keys():
            i2w[w2i[key]] = key
        return i2w

    def embedding(self):
        tokens = list(self.glove.keys())
        words = list(set(tokens + list(self.w2i.keys())))
        self.embed = torch.zeros(len(words),len(self.glove['the']))
        print('size of embedding')
        print(len(words))
        n = self.VOCAB_SIZE
        #extend w2i based on new things
        difword = sorted(set(tokens).difference(set(list(self.w2i.keys()))))
        print('vocab size')
        print(n)
        print('len of difference words')
        print(len(list(difword)))
        t = n
    
        for word in list(difword):
            self.w2i[word] = t
            t+=1
        
        print(len(self.w2i))
        for token in tokens:
            self.embed[self.w2i[token]] = torch.tensor(self.glove[token])
        #for Chinese, do not divided by std, for english divide by std
        #self.embed /= torch.std(self.embed)
        self.emb_layer = nn.Embedding.from_pretrained(self.embed)
        self.emb_layer.weight.requires_grad = False

    #calculate the token frequency
    def frequency(self):
        self.wcounter = Counter(normalize(token.form) if token.form is not None else '_' for sent in self.data for token in sent)
        wcounter = Counter(token.form if token.form is not None else '_' for sent in self.data for token in sent)
        self.ccounter = Counter(c for token, freq in wcounter.items() for c in token)
        #self.wcounter = Counter(normalize(token.form) for sent in self.data for token in sent) 
        #self.ccounter = Counter(c for sent in self.data for token in sent for c in token.form)

    #change the sentence into id sequences, create the corresponding matrix of relationship
    def createTrainingCorpus(self):
        self.sent = []
        self.char = []
        self.arbores = []
        self.postag = []
        self.depreltag = []
        self.punct = []
        self.arc = []
        self.sib = []
        nword = 0
        for sent in self.data:
            #sentences to id sequence
            nsent = [token.form if token.form is not None else '_' for token in sent]
            self.sent.append([self.w2i[normalize(word)] if word in self.w2i.keys() else self.w2i['<unk>'] for word in nsent])
            #self.sent.append([self.w2i[normalize(token.form)] if normalize(token.form) in self.w2i.keys() else self.w2i['<unk>'] for token in sent])
            self.sent[-1].insert(0,self.w2i['<bos>'])
            #char to sequence
            self.punct.append([ispunct(normalize(token)) for token in nsent])
            self.char.append([[self.c2i[c] if c in self.c2i.keys() else self.c2i['<unk>'] for c in token] for token in nsent])
            
            #self.char[-1].insert(0,[self.c2i[c] if c in self.c2i.keys() else self.c2i['<unk>'] for c in '<bos>'])
            self.char[-1].insert(0,[self.c2i['<bos>']])
            #xpos to id sequence
            self.postag.append([self.xpos[token.upos.upper()] if not(token.upos==None) else self.xpos['*UNK*'] for token in sent])
            self.postag[-1].insert(0,self.xpos['*ROOT-POS*'])
            #deprel to id sequence
            #print([token.form for token in sent])
            self.depreltag.append([self.deprel[token.deprel] if token.deprel in self.deprel.keys() else self.deprel['*UNK*'] for token in sent])
            #self.depreltag.append([self.deprel[token.deprel] for token in sent])
            #create the matrix of arc
            temp = np.zeros([len(sent),len(sent)])
            for i in range(len(sent)):
                head = int(sent[i].head)
                if head == 0: temp[i][i] = 1
                else: temp[head-1][i] = 1
            self.arbores.append(temp)
            
            #construct siblings
            seq = [int(token.head) for token in sent]
            self.arc.append([0]+seq)
            sib = self.get_sibs(seq)
            self.sib.append(sib) 
            nword += len(self.sent[-1])-1
        print('number of tokens: ' + str(nword))

    #convert all the file into pytorch like form
    def torchConversion(self):
        self.torchSent = []

        if self.fix_len is None: self.fix_len = max(len(token) for sequence in self.char for token in sequence)
        self.torchChar = [torch.tensor([ids[:self.fix_len] + [0] * (self.fix_len - len(ids)) for ids in sequence]) for sequence in self.char]

        self.torchPostag = []
        self.torchDepreltag = []
        self.torchArbores = []
        self.torchSib = []
        self.torchArc = []
        self.torchPunct = []
        for i in range(len(self.sent)):
            self.torchSent.append(torch.tensor(self.sent[i]))
            self.torchPostag.append(torch.tensor(self.postag[i]))
            self.torchDepreltag.append(torch.tensor(self.depreltag[i]))
            self.torchArbores.append(torch.tensor(self.arbores[i]))
            self.torchPunct.append(torch.tensor(self.punct[i]))
            self.torchSib.append(torch.tensor(self.sib[i]))
            self.torchArc.append(torch.tensor(self.arc[i]))
    
    def chunk(self, n_buckets =32, batch_size=5000):
        self.lengths = [sent.size(0)-1 for sent in self.torchSent]
        buckets = dict(zip(*utils.kmeans(self.lengths, n_buckets)))
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        self.chunks = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]
        
    #save the torch data into pickle form
    def save(self):
        if istraining:
            f = open('trainingSent.pickle','wb')
            pickle.dump(self.torchSent,f)
            f.close()
            f = open('trainingPostag.pickle','wb')
            pickle.dump(self.torchPostag,f)
            f.close()
            f = open('trainingArbores.pickle','wb')
            pickle.dump(self.torchArbores,f)
            f.close()
        else:
            f = open('ValSent.pickle','wb')
            pickle.dump(self.torchSent,f)
            f.close()
            f = open('ValPostag.pickle','wb')
            pickle.dump(self.torchPostag,f)
            f.close()
            f = open('ValArbores.pickle','wb')
            pickle.dump(self.torchArbores,f)
            f.close()

