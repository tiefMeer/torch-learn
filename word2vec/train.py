#!/usr/bin/env python3

# basic lib
import os
import re
import time
import jieba
import torch
import multiprocessing as mp

# local lib
from datasetLocal import ZHWIKI_AA

from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

def pretreat(text):
    li = list(jieba.cut(text))
    pattern = re.compile("[\u4e00-\u9fff0-9a-zA-Z]+")
    for word in li:
        if not pattern.fullmatch(word):
            li.remove(word)
    return li

# cutting words and building vocab lib
# change word to number, text to intlist
def yield_tokens(data_iter):
    for item in data_iter:
        yield pretreat(item["text"])

# 将一段字符串文本t数字化
def text_pipeline(t, hp):
    return torch.tensor(hp.vocab(pretreat(t))).to(hp.device)

# Hyper parameters for model and other genernal setting 
# instantiate before runing the model and after defining model
class HyperParameter():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.EPOCHS = 5
        self.BATCH_SIZE = 32
        self.LR = 10
        self.embed_dim = 100
        self.vocab_path = "data/corpus/vocab_AA.dat"
        self.model_path = "data/corpus/AA.mo"
        self.criterion = torch.nn.CrossEntropyLoss()
    def addVocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    def modelSetting(self, model_para):
        self.optimizer = torch.optim.SGD(model_para, lr=self.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)
            
# Define the model
class CBOWmodel(nn.Module):
    def __init__(self, hp, bagsize = 6):
        super(CBOWmodel, self).__init__()
        self.bag_size  = bagsize           # 词袋不包括预测词自己, 是输入词的数量
        self.embedding = nn.Embedding(hp.vocab_size, hp.embed_dim)
        self.rebound   = nn.Linear(hp.embed_dim, hp.vocab_size)
        self.sm        = nn.Softmax(dim=1)
        self.init_weights(hp)
        hp.modelSetting(self.parameters())
    def init_weights(self, hp):
        self.embedding.weight.data.uniform_(0, 1)
        self.rebound.weight.data.uniform_(0, 1)
        self.rebound.bias.data.zero_()
    def forward(self, wordBag):
        #avg_word_vec = sum(map(self.embedding, wordBag)) / self.bag_size
        #return torch.unsqueeze(self.sm(self.rebound(avg_word_vec)) , 0 )
        mid = self.embedding(wordBag)
        avg = torch.sum(mid, 0, keepdim=True) / self.bag_size
        return self.sm(self.rebound(avg)) 

def buildVocab(hp):
    if os.path.exists(hp.vocab_path):
        vocab = torch.load(hp.vocab_path)
    else:
        train_iter = ZHWIKI_AA()
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, hp.vocab_path)
    hp.addVocab(vocab)

# 用model作用于text上, 预测locate处的词
# 输入model:CBOWmodel, text:tensor数字化的一段文本, locate:index 数
# 返回多个词包的预测平均值
# hp暂时不用
'''
eef predit(text, locate, model, hp = None):
    bagsize = model.bag_size
    textsize = len(text)
    begin = locate - bagsize 
    end   =  begin + bagsize +1
    res = []
    while begin < locate:
        li = list( text[max(0, begin): locate] )
        li.extend( text[locate+1: end] )
        li.extend( [0] * (bagsize-len(li)) )
        res.append( model(torch.tensor(li)) )
        begin += 1
        end +=1
    return sum(res) / model.bag_size
'''

# 仅考虑locate 为中心的词袋
def predit(text, locate, model, hp):
    begin = locate - int(model.bag_size/2)
    end   =  begin + model.bag_size +1
    li = torch.cat( (text[max(0, begin): locate], text[locate+1: end]) )
    if model.bag_size-len(li) > 0:
        li = torch.cat((li, torch.tensor([0]*(model.bag_size-len(li))).to(hp.device)))
    return model(li)

# 针对一段数字化文本的训练
# 返回预测正确率
def train_one_text(text, model, hp):
    count_accu = 0
    for idx, word in enumerate(text):
        #batch = torch.unsqueeze( predit(text, idx, model, hp), 0 )
        batch =  predit(text, idx, model, hp)
        target = torch.tensor([word]).to(hp.device)
        loss = hp.criterion(batch, target)
        #loss.requires_grad_(True)
        hp.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        hp.optimizer.step()
        if torch.argmax(batch[0]) == target:
            count_accu += 1
        if idx == 10:
            print("debug: ")
            print("batch: ",batch)
            print("target: ",target)
            print("batch[target]: ",batch[0][target])
            print("loss: ", loss)
            print("accu: ", count_accu)
    return count_accu / len(text)

def train(data_iter, model, hp):
    model.train()
    print("文本编号\t正确率\t耗时")
    print('-'*30)
    res = []
    for idx, item in enumerate(data_iter):
        start_time = time.time()
        text = item["text"]
        one_accu = train_one_text(text_pipeline(text, hp), model, hp)
        res.append(one_accu)
        print(idx, '\t\t', one_accu, "\t{:8.3f}".format(time.time()-start_time) )
    return sum(res) / len(res)

'''
def evaluate(dataloade, embed_dim=64r, model):
    moutputodel.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            output = predit(item["text"])
            loss = criterion(output, )
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
'''

def main():
    hp = HyperParameter()
    buildVocab(hp)
    model = CBOWmodel(hp).to(hp.device)
    print(list(jieba.cut("我和这个是以恶搞")))

    # 开始训练
    train_iter = ZHWIKI_AA()
    for epoch in range(1, hp.EPOCHS + 1):
        epoch_start_time = time.time()
        avg_accu = train(train_iter, model, hp)
        print('#' * 50)
        print("# epoch: {:3d} | time: {:5.2f}s | 正确率: {:8.3f} ".format(
                epoch, time.time()-start_time, avg_accu) )
        print('#' * 50)

    torch.save(model, hp.model_path)

if __name__ == "__main__":
    main()
