import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from modules import (MLP, MLP_static, MLP_const, MLP_convex, MLP_convex_static, Triaffine, Biaffine, Biaffine_convex, BiLSTM, CHAR_LSTM, CHAR_CNN, CHAR_LSTM_D, SharedDropout, IndependentDropout)

from torch.distributions.gumbel import Gumbel

import time

from custom_rnn import CusRNNCell, CusRNNLayer, BiCusRNNLayer
#from biaffine import Biaffine

class GlobalEnergyModel(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, POS_SIZE, DEPREL_SIZE, bert=None, embed=0, xembed=None):
        
        super(GlobalEnergyModel,self).__init__()
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.POS_SIZE = POS_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        if bert is None: self.INPUT_SIZE = config.wemb_size + config.n_feat_embed + config.pemb_size
        else: self.INPUT_SIZE = config.wemb_size + config.n_feat_embed + 768
        self.membed=embed
        if embed==1: self.pretrained = xembed
        self.bert = bert
        self.gumbel = Gumbel(0, 1)
        ##common parts
        
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.pos_embed = nn.Embedding(POS_SIZE,config.pemb_size)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        
        
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        
        print('deprel size: ' + str(DEPREL_SIZE))
        #BILSTM
        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=config.lstm_layer,dropout=0.33)
        self.lstm_dropout = SharedDropout(p=0.33)

        self.mlp_arc_d = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_s = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_d = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_h = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=config.sib_mlp_size, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)


    #initialization of parameters
    def reset_parameters(self):
        nn.init.zeros_(self.W)
        nn.init.zeros_(self.U)

    def bertEmb(self):
        embBSent = self.bert(self.bsent)
        embBSent = embBSent.hidden_states[-4:]
        embBSent = [temp.unsqueeze(0) for temp in embBSent]
        embBSent = torch.cat(embBSent, 0)
        embBSent = embBSent.mean(0)
        return embBSent
    """
    #The first level of features by a BILSTM
    def BILSTMFeatures(self,embSent,embBSent,embChar,lengths):

        if embBSent is not None: emb = torch.cat((embSent,embBSent,embChar),2)
        else: emb = torch.cat((embSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        Lfeatures, _ = self.lstm(pack_emb)
        #Lfeatures = self.LSTM(pack_emb)
        pad_emb,_ = pad_packed_sequence(Lfeatures,batch_first=True,total_length=lengths.max()+1)
        pad_emb = self.lstm_dropout(pad_emb)
        return pad_emb
    """
    #The first level of features by a BILSTM
    def BILSTMFeatures(self,embSent,embBSent,embPos,embChar,lengths):
        
        if embBSent is not None: emb = torch.cat((embSent,embBSent,embPos,embChar),2)
        else: emb = torch.cat((embSent,embPos,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        Lfeatures, _ = self.lstm(pack_emb)
        #Lfeatures = self.LSTM(pack_emb)
        pad_emb,_ = pad_packed_sequence(Lfeatures,batch_first=True,total_length=lengths.max()+1)
        pad_emb = self.lstm_dropout(pad_emb)
        return pad_emb
    
    #Linear model features
    def LinearFeatures(self, features, max_length):
        Heads = features
        Modifiers = features
        #Modifiers = features[:,1:,:]
        return Heads, Modifiers

    #energy matrix
    def LinearEnergy(self, head, modifier, batch_size, max_length):
        L1H = self.mlp_arc_h(head)
        L1M = self.mlp_arc_d(modifier)
        #E = L1H @ self.W @ L1M.transpose(-1,-2) + torch.matmul(L1H,self.U).unsqueeze(2)
        E = self.arc_attn(L1M,L1H)
        if self.training: E += self.gumbel.sample(E.size()).to(self.device)
        res = E[:,1:,1:].transpose(-1,-2)
        idx = torch.arange(0,max_length,dtype=torch.long,device=self.device)
        res[:,idx,idx]=E[:,1:,0]
        res=res.reshape(batch_size,max_length,max_length)
        return (E, res)

    #siblings
    def SibEnergy(self, feature, batch_size, max_length):
        sib_s = self.mlp_sib_s(feature)
        sib_d = self.mlp_sib_d(feature)
        sib_h = self.mlp_sib_h(feature)
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        return s_sib

    #rel matrix
    def LinearRel(self, head, modifier, batch_size, max_length):
        L1H = self.mlp_rel_h(head)
        L1M = self.mlp_rel_d(modifier)
        E = self.rel_attn(L1M,L1H).permute(0,3,2,1)
        if self.training: E += self.gumbel.sample(E.size()).to(self.device)
        res = E[:,1:,1:,:]
        idx = torch.arange(0,max_length,dtype=torch.long,device=self.device)
        res[:,idx,idx,:]=E[:,0,1:,:]
        #res = res.permute(0,2,3,1)
        return (res - torch.logsumexp(res,-1).unsqueeze(-1), res)

    def LinearEnergyMatrix(self, features, batch_size, max_length=None):
        Heads, Modifiers = self.LinearFeatures(features, max_length)
        return self.LinearEnergy(Heads, Modifiers, batch_size, max_length), self.LinearRel(Heads, Modifiers, batch_size, max_length), self.SibEnergy(features, batch_size, max_length)
    
    #for every new sentence, do once initialization
    def Initialization(self, sent, bsent, char, pos, deprel, tree, batch_size, lengths, mask, mask1, mask2):
        self.train()
        self.sent = sent
        self.bsent = bsent
        ext_mask = sent>=self.VOCAB_SIZE
        self.ext_sent = sent.masked_fill(ext_mask, 1)

        self.pos = pos
        #self.lengths, self.perm_index = lengths.sort(0,descending=True)
        self.lengths = lengths
        self.char = char

        self.deprel = deprel
        self.tree = tree
        self.mask = mask
        self.mask1 = mask1
        self.mask2 = mask2
        self.batch_size = batch_size
        self.max_length = lengths.max()
        embSent = self.word_embed(self.ext_sent)
        if self.membed==1: 
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        embPos = self.pos_embed(self.pos)
        feat_embed = self.feat_embed(self.char[self.mask2])
        embChar = pad_sequence(feat_embed.split((self.lengths+1).tolist()), True) 
        embSent, embPos, embChar = self.embed_dropout(embSent,embPos,embChar)
        #embSent, embChar = self.embed_dropout(embSent,embChar)
        if self.bert is not None: embBSent = self.bertEmb()
        else: embBSent = None
        #self.LSTMfeatures = self.BILSTMFeatures(embSent,embBSent,embChar,self.lengths)
        self.LSTMfeatures = self.BILSTMFeatures(embSent,embBSent,embPos,embChar,self.lengths)
        self.features = self.LSTMfeatures
        (self.E, self.linearMatrix), (self.logRel, self.Erel), self.E_sib = self.LinearEnergyMatrix(self.features, self.batch_size, self.max_length)


    #for every new sentence, do once initialization(evaluation)
    def EVAInitialization(self, sent, bsent, char, pos, batch_size, lengths, mask, mask1, mask2):
        self.eval()
        self.sent = sent
        self.bsent = bsent
        ext_mask = sent>=self.VOCAB_SIZE
        self.ext_sent = sent.masked_fill(ext_mask, 1)
        self.pos = pos
        self.batch_size = batch_size
        #self.lengths, self.perm_index = lengths.sort(0,descending=True)
        self.lengths = lengths
        self.max_length = lengths.max()
        self.char = char
        self.mask = mask
        self.mask1 = mask1
        self.mask2 = mask2
        self.max_length = lengths.max()
        embSent = self.word_embed(self.ext_sent)
        if self.membed==1: 
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        embPos = self.pos_embed(self.pos)
        feat_embed = self.feat_embed(self.char[self.mask2])
        embChar = pad_sequence(feat_embed.split((self.lengths+1).tolist()), True)
        if self.bert is not None: embBSent = self.bertEmb()
        else: embBSent = None
        #self.LSTMfeatures = self.BILSTMFeatures(embSent,embBSent,embChar,self.lengths)
        self.LSTMfeatures = self.BILSTMFeatures(embSent,embBSent,embPos,embChar,self.lengths)
        self.features = self.LSTMfeatures
        (self.E, self.linearMatrix), (self.logRel, self.Erel), self.E_sib = self.LinearEnergyMatrix(self.features, self.batch_size, self.max_length)

    
