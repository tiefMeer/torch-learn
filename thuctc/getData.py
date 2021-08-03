#!/usr/bin/env python

import os
import gc
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
from torchtext.data.datasets_utils import _RawTextIterableDataset
import matplotlib.pyplot as plt
from meta import * 

#fields = glob.glob("data/*")
fields= ["data/科技", "data/时政"]

def getDataIter(field):
    for item in glob.glob(field+"/*"):
        res={}
        with open(item) as f:
            res["title"] = f.readline().strip()
            res["text"]  = f.read().strip()
        yield res

def dataset(field):
    NAME = field
    NUM_ITEMS = len(glob.glob(field+"/*.txt")) 
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter(field))

def loadModel(model_path=gp.model_path, name="Model"):
    if os.path.exists(model_path) and \
       os.path.getsize(model_path) != 0:
        gp.model[name] = torch.load(model_path)
    else:
        #last_model = max(glob.glob(model_path+".*"))
        last_model = ""     # 用于取最后一个模型备份．暂时不用
        if os.path.exists(last_model) and \
           os.path.getsize(last_model) != 0:
            gp.model[name] = torch.load(last_model)
        else:
            Model = eval(name)
            gp.model[name] = Model().to(gp.device)  # 不存在现有模型，新建一个
    #gp.modelSetting(gp.model[name].parameters())
    return gp.model[name]

def compression(unpack, layer):
    return res

def takeout(unpack):
    batch_size, maxlen, out_dim = unpack[0].shape   # batch_first=True
    res = torch.zeros(batch_size, out_dim)
    for n,length in enumerate(unpack[1]):
        res[n] = unpack[0][n][length-1]
    return res.to(gp.device)

class EncodeModel(nn.Module):
    def __init__(self):
        super(EncodeModel, self).__init__()
        # layers
        self.embedding = nn.Embedding(gp.vocab_size, gp.embed_dim)
        self.gru = nn.GRU(gp.embed_dim, gp.hidden_dim, batch_first=True)
        self.out = nn.Linear(gp.hidden_dim, gp.vocab_size)
        self.lsm = nn.LogSoftmax(dim=1)
        # optim
        self.LR = gp.defaultLR
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.LR)
    def initPara(self):
        pass
    def initHidden(self):
        return 
    def forward(self, inputs):
        print(inputs)
        text, length = inputs              # text:<batch_size x seq_len>
        embed = self.embedding(text)       # embed: <batch_size x seq_len x embed_dim> 
        gru_in = pack_padded_sequence(embed, length, batch_first=True) 
        gru_out, hidden = self.gru(gru_in)
        unpack = pad_packed_sequence(gru_out, batch_first=True)
        #output = self.lsm( self.out(takeout(unpack)) )
        output = unpack[0]
        return output, hidden
    
class AttnDecodeModel(nn.Module):
    def __init__(self, **para):
        #para
        self.embed_dim = gp.embed_dim
        self.hidden_dim = gp.hidden_dim
        self.output_size = gp.vocab_size
        self.max_length = 20
        self.dropout_p = 0.01
        super(AttnDecodeModel, self).__init__()
        # layers
        self.embedding = nn.Embedding(self.output_size, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.sm = nn.Softmax(dim=2)
        self.attn = nn.Linear(self.embed_dim + self.hidden_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.output_size)
        # optim
        self.LR = gp.defaultLR
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.LR)
    def initPara(self):
        pass
    def forward(self, inputs):
        # prev_words: <batch_size>
        prev_words, hidden, encoder_outputs = inputs
        print("inputs:\n", inputs)
        print("prev_words: ", prev_words.shape)
        print("hidden: ", hidden.shape)
        hidden = hidden.permute(1,0,2)       # 因为有batch_first=True, 此处不应该有这个
        print("new hidden: ", hidden.shape)
        print("encoder_outputs: ", encoder_outputs.shape)
        embed = self.embedding(prev_words)   # <batch_size x 1 x embed_dim>
        embed = self.dropout(embed)          # <batch_size x 1 x embed_dim>
        print("embed_drop: ", embed.shape)
        union = torch.cat((embed, hidden), 2)
        print("union: ", union.shape)
        attn_w = self.sm(self.attn(union))
        print("attn_weights: ", attn_w.shape)
        score = torch.bmm(attn_w, encoder_outputs)
        print("score: ", score.shape)
        output = torch.cat((embed[0], score[0]), 1)
        print("o1", output.shape)
        output = self.attn_combine(output).unsqueeze(0)
        print("o2", output.shape)
        output = self.relu(output)
        print("o3", output.shape)
        output, hidden = self.gru(output, hidden)
        print("o4", output.shape)
        output = self.out(output[0])
        print("o5", output.shape)
        return output, hidden

def makeInputs(item, *args, **kargs):
    return item[0], item[1]

def makeTarget(item, *args, **kargs):
    return item[2], item[3] 

def seqLoss():
    loss = 0
    return loss

# inputs: 数字化文本序列包,带长度
# target: 数字化标题序列包,带长度
# batch_first:False shape:<序列长度 x batch_size>
def seq2seqTrainUnit(inputs, target, gp, debug=False):
    print("inputs:\n", inputs)
    print("target:\n", target)
    loss = 0
    encoder,decoder = gp.model["EncodeModel"], gp.model["AttnDecodeModel"]
    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()

    #encode_output: <batch_size x vocab_size>
    #encode_hidden: <1 x batch_size x hidden_dim>
    encoder_outputs, encoder_hidden = encoder(inputs)
    
    seq_start = torch.fill_(torch.zeros(gp.BATCH_SIZE, 1), 
                            gp.vocab["<SOS>"]).to(gp.device).int()
    decoder_inputs = (seq_start, encoder_hidden, encoder_outputs)
    for di in range(max(target[1])):
        decoder_output, decoder_hidden = decoder(decoder_inputs)
        topv, topi = decoder_output.topk(1)
        print("decode_output:", decoder_output.shape, "\n", decoder_outputs)
        print("topv: \n", topv, "topi:\n", topi)
        input()
        decoder_inputs = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target[di])
        if decoder_inputs.item() == EOS_token:
            break

    print("loss: ", loss)
    loss.backward()
    encoder.optimizer.step()
    decoder.optimizer.step()
    #print("output:\n", output)
    return output, rightPredictionCount(output, target), loss

def seq2seqTrain(dataloader, gp):
    total_acc, total_count = 0, 0
    lossList = []
    gp.model["EncodeModel"].train()
    gp.model["AttnDecodeModel"].train()
    for idx, item in enumerate(dataloader):
        print("batch num: ", total_count)
        output, acc, loss = seq2seqTrainUnit(makeInputs(item), makeTarget(item), gp)
        print("acc:", acc)
        total_acc += acc
        total_count += gp.BATCH_SIZE
        lossList.append(loss)
    avg_accu = total_acc/total_count 
    return avg_accu, lossList 

def collate_fn(batch):
    interim = []
    for dic in batch:
        text = text_pipeline(dic["text"])
        title = text_pipeline(dic["title"])
        interim.append( [text, len(text)+1, title, len(title)+1] ) # +1是下面的<SOS>
    interim.sort(key=lambda x: x[1], reverse=True)
    text, text_length, title, title_length = list( zip(*interim) )

    text = pad_sequence(text, batch_first=True)
    adding = torch.fill_(torch.zeros(text.size(0), 1), gp.vocab["<SOS>"]).to(gp.device)
    text = torch.cat((adding.int() , text), dim=1)  # fill_()会变float，所以加int()

    title = pad_sequence(title, batch_first=True)
    adding = torch.fill_(torch.zeros(title.size(0), 1), gp.vocab["<SOS>"]).to(gp.device)
    title = torch.cat((adding.int() , title), dim=1)  # fill_()会变float，所以加int()

    return (text.to(gp.device), torch.tensor(text_length), 
            title.to(gp.device), torch.tensor(title_length)  )

def getDataloader(dataset=dataset):
    data = dataset(fields[0])
    return DataLoader(data, 
                      batch_size=gp.BATCH_SIZE, 
                      collate_fn=collate_fn)

def run():
    buildVocab(lambda: getDataIter(fields[0]))
    loadModel(gp.encode_model_path, name="EncodeModel")
    loadModel(gp.decode_model_path, name="AttnDecodeModel")

    for epoch in range(1, gp.EPOCHS+1):
        start_time = time.time()
        print('#' * 50)
        dataloader = getDataloader()
        avg_accu, lossList = seq2seqTrain(dataloader, gp)
        updateLR()
        print('#' * 50)
        print("# epoch: {:3d} | time: {:5.2f}s | avg_accu: {:8.3f} | nextLR: {:8.3f} ".\
                format(epoch, time.time()-start_time, avg_accu, gp.LR) )
        #plt.plot(lossList)
        #plt.savefig(gp.fig_path+"lossList-"+str(time.time_ns())+".png")
        gc.collect()
    torch.save(model, gp.model_path)
    #torch.save(model, gp.model_path +"."+ str(time.time_ns()) +".bak")


run()

