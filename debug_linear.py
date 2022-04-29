import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import argparse
import random
import numpy as np
import time
import pickle
import os
import copy
from transformers import BertTokenizer,BertModel

from ext_emb import load, create_emb_layer, create_emb_layer_
from preprocessing import Preprocessing
from parser_linear import GlobalEnergyModel
from model_linear import model
from config import Config
import utils


if __name__ == '__main__':
   
    #set random seed for every run
    random.seed(time.time())
    torch.manual_seed(random.randint(0,999999))
    np.random.seed(random.randint(0,999999))
    
    #torch.autograd.set_detect_anomaly(True)
    #set parameters for training
    userParser = argparse.ArgumentParser()
    userParser.add_argument("--gpu", "-g", type=int, default=1, choices=[0,1,2,3,4], help="the number of gpu to use")
    userParser.add_argument("--train", "-t", default='/users/xudong.zhang/data/ud-treebanks-v2.0/UD_Chinese/zh-ud-train.conllu', help="path to training data")
    userParser.add_argument("--dev", "-d", default='/users/xudong.zhang/data/ud-treebanks-v2.0/UD_Chinese/zh-ud-dev.conllu', help="path to dev data")
    userParser.add_argument("--test", '-u', default='/users/xudong.zhang/data/ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/zh.conllu', help='path to test data')
    userParser.add_argument('--batch', '-b', type=int, default=32, help="batch size")
    userParser.add_argument('--epoch', '-e', type=int, default=30)
    userParser.add_argument('--config', '-c', default='config.ini', help='path to config file')
    userParser.add_argument('--activation', '-a', default='PReLU', help='activation function')
    userParser.add_argument('--mode', '-m', default='Linear', help='mode of training (Linear, Global, General)')
    userParser.add_argument('--preTrained', '-p', default='./modelLinearBert11/', help='Pretrained Part')
    userParser.add_argument('--save', '-s', default='./modelLinearBert/', help='Folder to save the files')
    userParser.add_argument('--rnn', '-r', type=int, default=0, choices=[0,1], help='Use RNN or not (1 yes, 0 no)')
    userParser.add_argument('--dropout','-o',type=int,default=1,choices=[0,1],help='For global part, use dropout or not (1 yes, 0 no)')
    userParser.add_argument('--loss', '-l',type=int,default=2,choices=[0,1,2],help='Loss function choice for linear part (0 hinge loss, 1 normalized hinge loss, 2 cross entropy)')
    userParser.add_argument('--xembed', '-x', type=int, default=1,choices=[0,1],help='Using external embedding or not (0 no, 1 yes)')
    #userParser.add_argument('--yembed', '-y', default='../glove.6B.100d.txt', help='path to external embedding')
    userParser.add_argument('--yembed', '-y', default='../../data/depretrained.txt', help='path to external embedding')

    args = userParser.parse_args()
    #tokenizer = BertTokenizer.from_pretrained('../BertEn/')
    glove = None 
    if args.xembed==1: glove = load(args.yembed)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    #train_data = Preprocessing(args.train,tokenizer=tokenizer,glove=glove)
    train_data = Preprocessing(args.train,glove=glove)
    emb_layer=None
    if args.xembed==1:
        emb_layer = train_data.emb_layer
    
    dev_data = Preprocessing(args.dev,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)
    test_data = Preprocessing(args.test,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)

    #dev_data = Preprocessing(args.dev,tokenizer=tokenizer,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)
    #test_data = Preprocessing(args.test,tokenizer=tokenizer,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)


    VOCAB_SIZE = train_data.VOCAB_SIZE
    CHAR_SIZE = train_data.CHAR_SIZE
    POS_SIZE = len(train_data.xpos)
    DEPREL_SIZE = len(train_data.deprel)
    
    
    config = Config(args.config)

    #bert = BertModel.from_pretrained('../BertEn/', output_hidden_states=True)

    #bert = bert.requires_grad_(False)

    bert = None

    num_batch = torch.tensor(train_data.chunks).sum().item()

    net = GlobalEnergyModel(device, config, VOCAB_SIZE, CHAR_SIZE, POS_SIZE, DEPREL_SIZE, bert, args.xembed, xembed=emb_layer).to(device=device) 
    
    """ 
    if args.mode=='Linear':
        #load the local model which gives the highest UAS on dev
        checkpoint = torch.load(args.preTrained+'DISLinear282.pth', map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        #exit()
        #del checkpoint
        #torch.cuda.empty_cache()  
    """
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),config.lr,(config.beta_1, config.beta_2),config.epsilon)
    scheduler = ExponentialLR(optimizer, config.decay ** (1 / config.steps))
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #del checkpoint
    torch.cuda.empty_cache()  
    Model = model(device, optimizer, scheduler, net)
    print('Successfully initialize network')    
    """
    test_uas, test_las = utils.evaluate(device,dev_data,Model,args.save+'pdev' + args.mode + str(114) + '.conll',0,114,train_data.ideprel)
    print(test_uas)
    print(test_las)
    test_uas, test_las = utils.evaluate(device,test_data,Model,args.save+'ptest' + args.mode + str(114) + '.conll',0,114,train_data.ideprel)
    print(test_uas)
    print(test_las)
    exit()
    """
    sentNum = 0.0
    sumLoss = 0.0
    iter_time = 0.0
    old_uas = 0.0
    old_las = 0.0
    old_test_uas = 0.0
    t_patience = 0
    index_best = 0
    #start the loop for training
    for ITER in range(args.epoch):
        epoch_loss = 0.0
        epoch_num = 0.0

        #for i in range(num_batch):
        range_fn = torch.randperm
        #range_fn = torch.arange
        t=1
        for i in range_fn(len(train_data.buckets)).tolist():
            split_sizes = [(len(train_data.buckets[i]) - j - 1) // train_data.chunks[i] + 1
                           for j in range(train_data.chunks[i])]
            for batch in range_fn(len(train_data.buckets[i])).split(split_sizes):
                indexs = [train_data.buckets[i][j] for j in batch.tolist()]
            #break
                start_time = time.time()
                optimizer.zero_grad()
                sent = [train_data.torchSent[index] for index in indexs]
                #bsent = [train_data.torchBSent[index] for index in indexs]
                char = [train_data.torchChar[index] for index in indexs]
                pos = [train_data.torchPostag[index] for index in indexs]
                deprel = [train_data.torchDepreltag[index] for index in indexs]
                structure = [train_data.torchArbores[index] for index in indexs]
                batch_size = len(sent)
                pad_sent,pad_char,pad_pos,pad_deprel,pad_structure,lengths, mask, mask1, mask2 = utils.padding_train(sent,char,pos,deprel,structure,batch_size,train_data.w2i,train_data.i2w,train_data.xpos)
                print('size of batch')
                print(lengths.sum())
                pad_sent = pad_sent.to(device)
                #pad_bsent = pad_bsent.to(device)
                pad_char = pad_char.to(device)
                pad_pos = pad_pos.to(device)
                pad_deprel = pad_deprel.to(device)
                pad_structure = pad_structure.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                mask1 = mask1.to(device)
                mask2 = mask2.to(device)
                
                pad_bsent = None
                
                loss = Model.train(pad_sent, pad_bsent,pad_char, pad_pos, pad_deprel, pad_structure, mask, mask1, mask2, batch_size, lengths)
                
                epoch_loss += loss
                
                epoch_num += 1.0
                
                sentNum+=1.0
                sumLoss+=loss
                delta_time = time.time()-start_time
                iter_time += delta_time
                print('batch: {}/{} avg.loss: {} loss: {} avg.time: {} time: {}'.format(t, num_batch, sumLoss/sentNum,loss,iter_time / sentNum,delta_time), flush=True,end='\n')
                t+=1
        print('finish epoch ' + str(ITER) + ' epoch loss: ' + str(epoch_loss/epoch_num),flush=True)


        #start to evaluate over the training data and dev data
        test_uas, test_las, test_uas_, test_las_ = utils.evaluate(device,dev_data,Model,args.save+'ptest' + args.mode + str(ITER) + '.conll',old_test_uas,ITER,train_data.ideprel)
        print('UAS for iterations ' + str(ITER) + '=' + str(test_uas))
        print('LAS for iterations ' + str(ITER) + '=' + str(test_las))
        print('UAS_ for iterations ' + str(ITER) + '=' + str(test_uas_))
        print('LAS_ for iterations ' + str(ITER) + '=' + str(test_las_))
        
        if test_las_ > old_las:
            if index_best!=0:
                os.remove(args.save+'DIS'+args.mode+str(index_best)+'.pth')
            index_best = ITER
            old_uas = test_uas_
            old_las = test_las_
            t_patience = 0

            torch.save({'model_state_dict': Model.net.state_dict(),'optimizer_state_dict': Model.optimizer.state_dict(),'scheduler_state_dict': Model.scheduler.state_dict()}, args.save+'DIS'+args.mode+str(ITER)+'.pth')
        else: 
            t_patience += 1
        if t_patience > 100: break
        #make regularization smaller
        test_uas, test_las, test_uas_, test_las_ = utils.evaluate(device,test_data,Model,args.save+'ptest' + args.mode + str(ITER) + '.conll',old_test_uas,ITER,train_data.ideprel) 
        print('Test UAS for iterations ' + str(ITER) + '=' + str(test_uas))
        print('Test LAS for iterations ' + str(ITER) + '=' + str(test_las))
        print('Test UAS_ for iterations ' + str(ITER) + '=' + str(test_uas_))
        print('Test LAS_ for iterations ' + str(ITER) + '=' + str(test_las_))
        

    print('All bugs fixed!')


