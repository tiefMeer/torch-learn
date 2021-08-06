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

fields = list(map(lambda s:"data/THUCNews/"+s, os.listdir("data/THUCNews/")))
#fields= ["data/THUCNews/科技", "data/时政"]

#dig:1d-list-like
def digital2text(dig):
    return reduce(lambda s1,s2: s1+s2, 
                  map(lambda i:gp.vocab.vocab.lookup_token(i), 
                      dig))

def getDataIter(field):
    for item in glob.glob(field+"/*"):
        res={}
        with open(item) as f:
            res["title"] = f.readline().strip()
            res["text"]  = f.read().strip()[:10000]
        yield res

def dataset(field):
    NAME = field
    NUM_ITEMS = len(glob.glob(field+"/*.txt")) 
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter(field))

def compression(unpack, layer):
    return res

def takeout(unpack):
    batch_size, maxlen, out_dim = unpack[0].shape   # batch_first=True
    res = torch.zeros(batch_size, out_dim)
    for n,length in enumerate(unpack[1]):
        res[n] = unpack[0][n][length-1]
    return res.to(gp.device)

def makeInputs(item, *args, **kargs):
    return [item[0], item[1]]

def makeTarget(item, *args, **kargs):
    return [item[2], item[3]]

def seqLoss():
    loss = 0
    return loss

def rightPredictionCount(pred, target):
    return pred.numel()-(pred-target).count_nonzero()

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

# inputs: 数字化文本序列包,带长度
# target: 数字化标题序列包,带长度
# batch_first:False shape:<序列长度 x batch_size>
def seq2seqTrainUnit(inputs, target, gp, debug=False):
    gp.optimizer.zero_grad()
    pred, loss = gp.model(inputs, target)
    acc = rightPredictionCount(pred, target[0])
    print("loss: ", int(loss))
    print("acc: ", int(acc))
    print("<<< ", digital2text(target[0][0]))
    print(">>> ", digital2text(pred[0]))
    loss.backward()
    gp.optimizer.step()
    return pred, acc, loss

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
        #if idx > 200:
        #    input()
        #print("acc:", acc)
        total_acc += acc
        total_count += gp.BATCH_SIZE
        lossList.append(loss)
    avg_accu = total_acc/total_count 
    return avg_accu, lossList 

def collate_fn(batch):
    interim = []
    for dic in batch:
        text = text_pipeline(dic["text"])
        title = title_pipeline(dic["title"])
        interim.append( [text, len(text)+1, title, len(title)+1] ) # +1是<SOS>或<EOS>
    interim.sort(key=lambda x: x[1], reverse=True)
    text, text_length, title, title_length = list( zip(*interim) )

    text = pad_sequence(text, batch_first=True)
    adding = torch.fill_(torch.zeros(text.size(0), 1), gp.vocab["<SOS>"]).to(gp.device)
    text = torch.cat((adding.int() , text), dim=1)  # fill_()会变float，所以加int()

    adding = torch.tensor([gp.vocab["<EOS>"]]).to(gp.device)
    title = list(map(lambda i:torch.cat((title[i], adding), dim=0), 
                     range(len(title))))
    title = pad_sequence(title, batch_first=True)
    #adding = torch.fill_(torch.zeros(title.size(0), 1), gp.vocab["<EOS>"]).to(gp.device)
    #title = torch.cat((adding.int() , title), dim=1)  # fill_()会变float，所以加int()

    return (text.to(gp.device), torch.tensor(text_length), 
            title.to(gp.device), torch.tensor(title_length)  )

def getDataloader(dataset=dataset):
    data = dataset(fields[0])
    return DataLoader(data, 
                      batch_size=gp.BATCH_SIZE, 
                      collate_fn=collate_fn)

def showEpoch(avg_accu, lossList, epoch_n, start_time):
    gp.epoch_loss_list.append(sum(lossList)/len(lossList))
    plt.cla()
    plt.plot(gp.epoch_loss_list)
    plt.savefig(gp.epoch_loss_list_fig_path)
    #plt.savefig(gp.epoch_loss_list_fig_path+str(time.time_ns())+".png")
    print('#' * 50)
    print("# epoch: {:3d} | time: {:5.2f}s | avg_accu: {:8.3f} | nextLR: {:8.3f} ".\
            format(epoch_n, time.time()-start_time, avg_accu, gp.LR) )
    plt.cla()
    plt.plot(lossList)
    plt.savefig(gp.fig_path+"lossList-"+str(time.time_ns())+".png")

def updateHyperParas():
    gp.BATCH_SIZE = 32 if gp.BATCH_SIZE == 64 else 64
    # ...

def runEpoch(epoch_n):
    start_time = time.time()
    print('#' * 50)
    dataloader = getDataloader()
    avg_accu, lossList = seq2seqTrain(dataloader, gp)
    showEpoch(avg_accu, lossList, epoch_n, start_time)
    #updateHyperParas()
    gc.collect()
    torch.save(gp.model, gp.model_path)
    if epoch_n%10 == 0:
        torch.save(gp.model, gp.model_path+"."+str(time.time_ns())+".bak")

### main ###
buildVocab(lambda: getDataIter(fields[0]))
loadModel()
if __name__ == "__main__":
    for e in range(1, gp.EPOCHS+1):
        runEpoch(e)


