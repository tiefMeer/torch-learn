#!/usr/bin/env python 

import os
import pdb
import glob
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)
import matplotlib.pyplot as plt
from count import *

import gc
import objgraph

def loadModel():
    if os.path.exists(gp.model_path) and \
       os.path.getsize(gp.model_path) != 0:
        gp.model = torch.load(gp.model_path)
    else:
        last_model = max(glob.glob(gp.model_path+".*"))
        if os.path.exists(last_model) and \
           os.path.getsize(last_model) != 0:
            gp.model = torch.load(last_model)
        else:
            gp.model = sentiModel().to(gp.device)
    gp.modelSetting(gp.model.parameters())
    return gp.model

def takeout(unpack):
    batch_size, maxlen, out_dim = unpack[0].shape
    res = torch.zeros(batch_size, out_dim)
    for n,length in enumerate(unpack[1]):
        res[n] = unpack[0][n][length-1]
    return res.to(gp.device)

class sentiModel(nn.Module):
    def __init__(self):
        super(sentiModel, self).__init__()
        self.embedding = nn.Embedding(gp.vocab_size, gp.embed_dim)
        self.lstm = nn.LSTM(gp.embed_dim, gp.hidden_dim, batch_first=True)
        self.toSenti = nn.Linear(gp.hidden_dim, gp.target_size)
        #gp.modelSetting(self.parameters())
    def forward(self, inputs):
        sentences, length = inputs
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm( pack_padded_sequence(embeds, length, batch_first=True) )
        unpack = pad_packed_sequence(lstm_out, batch_first=True)
        res = self.toSenti( takeout(unpack) )
        #print("sentences:",sentences.shape,"\n",sentences)
        #print("length:\n",length)
        #print("embeds:\n",embeds)
        #print("lstm_out:\n",lstm_out)
        #print("unpack:",unpack[0].shape,"\n",unpack)
        #print("res:\n",res)
        return res

def collate_fn(batch):
    interim = []
    for dic in batch:
        interim.append( [dic["text"], len(dic["text"]), dic["senti"]] )
    interim.sort(key=lambda x: x[1], reverse=True)
    text, length, senti = list( zip(*interim) )
    #text, length = list(zip(*(sorted(zip(text, length),
    #                                 key=lambda x:x[1], reverse=True))))
    text = pad_sequence(text, batch_first=True)
    return (text.to(gp.device), 
            torch.tensor(length), 
            torch.tensor(senti).to(gp.device) )

def rightPredictionCount(output, target):
    all = len(target)
    #print(all)
    #print(torch.argmax(output, dim=1))
    res = all - torch.count_nonzero(torch.argmax(output, dim=1)-target)
    return res

def makeInput(item, *args, **kargs):
    return item[0], item[1]

def makeTarget(item, *args, **kargs):
    return item[2].to(gp.device)

def trainUnit(inputs, target, model, gp, debug=False):
    gp.optimizer.zero_grad()
    output = model(inputs)
    loss = gp.criterion(output, target)
    #print("output:\n", output)
    #print("target:\n")
    #print(target)
    print("loss: ", loss)
    loss.backward()
    gp.optimizer.step()
    return output, rightPredictionCount(output, target), loss

def train(dataloader, model, gp, debug=False):
    total_acc, total_count = 0, 0
    lossList = []
    #hid = model.initHidden()
    model.train()
    for idx, item in enumerate(dataloader):
        print("batch: ", total_count)
        output, acc, loss = trainUnit(makeInput(item), makeTarget(item), model, gp)
        print("acc:", acc)
        total_acc += acc
        total_count += 1 
        lossList.append(loss)
    avg_accu = total_acc/total_count 
    return avg_accu, lossList

def getDataloader():
    return DataLoader(labledDataset(), batch_size=gp.BATCH_SIZE, 
                                       collate_fn=collate_fn)

def updateLR():
    f  = lambda x:math.log(1+x)
    f8 = lambda x:f(f(f(f(f(f(f(f(x))))))))
    gp.LR = f8(gp.LR*5)/5

def run():
    buildVocab(getDataIter)
    model= loadModel()
    for epoch in range(1, gp.EPOCHS+1):
        start_time = time.time()
        print('#' * 50)
        dataloader = getDataloader()
        avg_accu, lossList = train(dataloader, model, gp)
        updateLR()
        print('#' * 50)
        print("# epoch: {:3d} | time: {:5.2f}s | avg_accu: {:8.3f} | nextLR: {:8.3f} ".\
                format(epoch, time.time()-start_time, avg_accu, gp.LR) )
        #plt.plot(lossList)
        #plt.savefig(gp.fig_path+"lossList-"+str(time.time_ns())+".png")
        gc.collect()
    torch.save(model, gp.model_path)
    #torch.save(model, gp.model_path +"."+ str(time.time_ns()) +".bak")


if __name__ == "__main__":
    try:
        run()
    except Exception as ex:
        print("ERROR:", ex)
        pdb.set_trace()
