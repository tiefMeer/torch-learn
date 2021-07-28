#!/usr/bin/env python

from train import *

class testModel(torch.nn.Module):
    def __init__(self, hp):
        super(testModel,self).__init__()
        self.embedding = nn.Embedding(10000, 100)
        self.rebound   = nn.Linear(100, 100)
        self.sm        = nn.Softmax(dim=1)
        self.init_weights(hp)
        hp.modelSetting(self.parameters())
    def init_weights(self, hp):
        self.rebound.weight.data.uniform_(0, 1)
        self.rebound.bias.data.zero_()
    def forward(self, input_):
        input_ = self.embedding(input_)
        input_ = torch.sum(input_, 0, keepdim=True) / len(input_)
        input_ = self.rebound(input_)
        return self.sm(input_)

# 采用testModel
def test1():
    hp = HyperParameter()
    buildVocab(hp)
    model = testModel(hp).to("cuda")
    train_iter = ZHWIKI_AA()
    #t=torch.rand(1,100)
    t=(torch.rand(15)*10000).int().to("cuda")
    print(t) 

    for i in range(200):
        pred = model(t)
        print(f"#{i} predit: ", pred)
    #    print(">>>\t", torch.argmax(pred))
        loss=hp.criterion(pred, torch.tensor([5]).to("cuda"))
        print(">>>loss:\t",loss)
        print(">>>pred:\t", pred[0][5])
        hp.optimizer.zero_grad()
        loss.backward()
        print("loss.item(): ", loss.item())
        hp.optimizer.step()


# 采用CBOWmodel
def test0():
    hp = HyperParameter()
    buildVocab(hp)
    model = CBOWmodel(hp).to("cuda")
    train_iter = ZHWIKI_AA()

    #t=torch.rand(1,100)
    #t=(torch.rand(15)*10000).int()
    t=text_pipeline(next(train_iter)["text"], hp).to("cuda")
    print(t[:20]) 

    for i in range(1000):
        pred = predit(t, 9, model, hp)
        print(f"#{i} predit: ", pred)
    #    print(">>>\t", torch.argmax(pred))
        loss=hp.criterion(pred, torch.tensor([9]).to("cuda"))
        print(">>>loss:\t",loss)
        print(">>>pred:\t", pred[0][9])
        hp.optimizer.zero_grad()
        loss.backward()
        hp.optimizer.step()


test1()


