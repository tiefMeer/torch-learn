#!/usr/bin/env python

import os
import gc
import sys
import pdb
import glob
import math
import time
from functools import reduce
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.datasets_utils import _RawTextIterableDataset

from model import * 

#dig:1d-list-like
def digital2text(dig):
    return reduce(lambda s1,s2: s1+s2, 
                  map(lambda i:gp.vocab.vocab.lookup_token(i), 
                      dig))

#def getDataIter(fname="data.txt"):
#    with open(fname) as f:
#        line = f.readline().strip()
#        while line:
#            yield line
#            line = f.readline().strip()
def getDataIter(fname="data.txt"):
    for i in range(1100, 1356):
        yield gp.vocab.vocab.lookup_token(i)

def dataset(fname="data.txt"):
    NAME="title"
    NUM_ITEMS=8425
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter(fname))

#def compression(unpack, layer):
#    return res

def takeout(unpack):
    batch_size, maxlen, out_dim = unpack[0].shape   # batch_first=True
    res = torch.zeros(batch_size, out_dim)
    for n,length in enumerate(unpack[1]):
        res[n] = unpack[0][n][length-1]
    return res.to(gp.device)

# 将很长的text的rnn的out挑选出gp.max_length个
# 使之方便attention对齐
def selectOut(en_out):
    res = torch.zeros(en_out.size(0), 0, gp.hidden_dim).to(gp.device)
    place = 0
    step = math.floor(en_out.size(1) / gp.max_length)
    for i in range(0, gp.max_length):
        res = torch.cat((res, en_out[:, place,:].unsqueeze(1)), dim=1)
        place += step
    return res

def rightPredictionCount(pred, target):
    return pred.numel()-(pred-target).count_nonzero()

def makeInputs(item, *args, **kargs):
    return [item[0], item[1]]

def makeTarget(item, *args, **kargs):
    return [item[2], item[3]]

def greatorMask(current_n, lengths):
    mask = current_n < lengths
    return mask.to(gp.device)

# pred_prob: <b x maxlen x vocab_size>
def calcLoss(pred_prob, target):
    loss = 0
    for di in range(max(target[1])):
    #for di in range(2):
        #print(target[0][:, di+1])
        loss_tensor = gp.criterion(pred_prob[:,di,:], target[0][:, di+1]) #+1是SOS 
        #loss += loss_tensor.masked_select(greatorMask(di, target[1])).mean()
        loss += loss_tensor.mean()
        #print("loss: ", loss)
    return loss

# inputs: 数字化文本序列包,带长度
# target: 数字化标题序列包,带长度
# batch_first:False shape:<序列长度 x batch_size>
def seq2seqTrainUnit(inputs, target, gp, debug=False):
    pred, pred_prob = gp.model(inputs)
    loss = calcLoss(pred_prob, target)
    acc = rightPredictionCount(pred, target[0])
    print("loss: ", int(loss))
    print("acc: ", int(acc))
    print("<<< ", digital2text(target[0][0]))
    print(">>> ", digital2text(pred[0]))
    gp.optimizer.zero_grad()
    loss.backward()
    gp.optimizer.step()
    return pred, acc, loss.item()

def seq2seqTrain(dataloader, gp):
    total_acc, total_count = 0, 1
    lossList = []
    gp.model.train()
    for idx, item in enumerate(dataloader):
        if len(item[0]) != gp.BATCH_SIZE:
            break
        print("batch num: ", total_count)
        output, acc, loss = seq2seqTrainUnit(makeInputs(item), makeTarget(item), gp)
        #print(int(total_count), "\t", int(loss), file=sys.stderr)
        #if idx > 20:
        #    input()
        #print("acc:", acc)
        total_acc += acc
        total_count += gp.BATCH_SIZE
        lossList.append(loss)
    avg_accu = total_acc/total_count 
    return avg_accu, lossList 

def collate_fn(batch):
    interim = []
    for line in batch:
        text = text_pipeline(line)
        interim.append( [text, len(text)+1] ) # +2是<SOS>和<EOS>
    interim.sort(key=lambda x: x[1], reverse=True)
    text, text_length = list( zip(*interim) )

    adding = torch.tensor([gp.vocab["<EOS>"]]).to(gp.device)
    text = list(map(lambda i:torch.cat((text[i], adding), dim=0), 
                     range(len(text))))
    text = pad_sequence(text, batch_first=True)
    adding = torch.fill_(torch.zeros(text.size(0), 1), gp.vocab["<SOS>"]).to(gp.device)
    text = torch.cat((adding.int() , text), dim=1)  # fill_()会变float，所以加int()

    length = torch.tensor(text_length)      #length需要是cpu tensor
    return (text.to(gp.device), length,     #inputs
            text.to(gp.device), length )    #target

def getDataloader(dataset=dataset):
    data = dataset()
    return DataLoader(data, 
                      batch_size=gp.BATCH_SIZE, 
                      collate_fn=collate_fn)

def showEpoch(avg_accu, lossList, epoch_n, start_time):
    gp.epoch_loss_list.append(sum(lossList)/len(lossList))
    plt.cla()
    plt.plot(gp.epoch_loss_list)
    plt.savefig(gp.epoch_loss_list_fig_path)
    #plt.savefig(gp.epoch_loss_list_fig_path+str(time.time_ns())+".png")
    plt.cla()
    plt.plot(lossList)
    plt.savefig(gp.fig_path+"lossList-"+str(time.time_ns())+".png")
    print('#' * 50)
    print("# epoch: {:3d} | time: {:5.2f}s | avg_accu: {:8.3f} | nextLR: {:8.6f} ".\
            format(epoch_n, time.time()-start_time, avg_accu, gp.LR) )

def updateHyperParas():
    gp.LR -= 0.00001
    gp.modelSetting()

def runEpoch(epoch_n):
    start_time = time.time()
    print('#' * 50)
    dataloader = getDataloader()
    avg_accu, lossList = seq2seqTrain(dataloader, gp)
    #updateHyperParas()
    showEpoch(avg_accu, lossList, epoch_n, start_time)
    gc.collect()
    torch.save(gp.model, gp.model_path)
    if epoch_n%10 == 0:
        torch.save(gp.model, gp.model_path+"."+str(time.time_ns())+".bak")

### main ###
buildVocab(lambda: getDataIter())
loadModel()
if __name__ == "__main__":
    for e in range(1, gp.EPOCHS+1):
        runEpoch(e)


