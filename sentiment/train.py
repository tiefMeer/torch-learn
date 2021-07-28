#!/usr/bin/env python 

import os
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

class sentiModel(nn.Module):
    def __init__(self):
        super(sentiModel, self).__init__()
        self.embedding = nn.Embedding(gp.vocab_size, gp.embed_dim)
        self.lstm = nn.LSTM(gp.embed_dim, gp.hidden_dim)
        self.toSenti = nn.Linear(gp.hidden_dim, gp.target_size)
        gp.modelSetting(self.parameters())
    def forward(self, inputs):
        sentences, length = inputs
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm( pack_padded_sequence(embeds, length) )
        res = self.toSenti( pad_packed_sequence(lstm_out)[0] )
        return res[-1]

def collate_fn(batch):
    text, length, senti = [], [], []
    for dic in batch:
        text.append(dic["text"])
        length.append(len(dic["text"]))
        senti.append(dic["senti"])
    #res.sort(key=lambda x: len(x[1]), reverse=True)
    text, length = list(zip(*(sorted(zip(text, length),
                                     key=lambda x:x[1], reverse=True))))
    text = pad_sequence(text)
    return torch.tensor(text), torch.tensor(length), torch.tensor(senti)

def isRightPrediction(output, target):
    return torch.equal(output, target)

def makeInput(item, *args, **kargs):
    return item[0], item[1]

def makeTarget(item, *args, **kargs):
    return item[2].to(gp.device)

def trainUnit(inputs, target, model, gp, debug=False):
    #hidden = (torch.rand(1,1,3), torch.rand(1,1,3))
    gp.optimizer.zero_grad()
    output = model(inputs)
    print("output:\n", output)
    print("target:\n", target)
    loss = gp.criterion(output, target)
    print("loss: ", loss)
    loss.backward()
    gp.optimizer.step()
    return output, isRightPrediction(output, target), loss

def train(dataloader, model, gp, debug=False):
    total_acc, total_count = 0, 0
    lossList = []
    #hid = model.initHidden()
    model.train()
    print(next(iter(dataloader)))
    for idx, item in enumerate(dataloader):
        print("item: ", item)
        output, acc, loss = trainUnit(makeInput(item), makeTarget(item), model, gp)
        total_acc += acc
        total_count += 1 
        lossList.append(loss)
    return total_acc/total_count, lossList
'''
#def run():
buildVocab(getDataIter)
model = sentiModel().to(gp.device)
gp.model = model
data = labledDataset()
dataloader = DataLoader(data, batch_size=gp.BATCH_SIZE, collate_fn=collate_fn)
for epoch in range(1, gp.EPOCHS+1):
    epoch_start_time = time.time()
    avg_accu, lossList = train(dataloader, model, gp)
    print('#' * 50)
    print("# epoch: {:3d} | time: {:5.2f}s | 正确率: {:8.3f} ".format(
            epoch, time.time()-start_time, avg_accu) )
    print('#' * 50)
    plt.plot(lossList)
    plt.show()
    torch.save(model, gp.model_path)
'''

