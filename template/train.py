#!/usr/bin/env python 

import torch
import torch.nn as nn

class TemplateModel(nn.Module):
    def  __init__():
        pass
    def init():
        pass
    def forword(inputs):
        pass

def isRightPrediction(output, target):
    return torch.equal(output, target)

def outputPipeline(output):
    return output

def makeInput(item, *args, **kargs):
    return item[0]

def makeTarget(item, *args, **kargs):
    return item[1]

def trainUnit(inputs, target, model, hp, debug=False):
    hp.optimizer.zero_grad()
    output = outputPipeline(model(inputs))
    loss = hp.criterion(output, target)
    loss.backward()
    hp.optimizer.step()
    return output, isRightPrediction(output, target), loss

def train(dataloader, model, hp, debug=False):
    total_acc, total_count = 0, 0
    model.train()
    for idx, item in enumerate(dataloader):
        total_acc += trainUnit(makeInput(item), makeTarget(item), model, hp)[1]
        total_count += 1  
    return total_acc/total_count

def RNNtrainUnit(inputs, target, model, hp, debug=False):
    hp.optimizer.zero_grad()
    for i in inputs:
        output = outputPipeline(model(inputs))
    loss = hp.criterion(output, target)
    loss.backward()
    hp.optimizer.step()
    return output, isRightPrediction(output, target), loss

def RNNtrain(dataloader, model, hp, debug=False):
    total_acc, total_count = 0, 0
    hid = model.initHidden()
    model.train()
    for idx, item in enumerate(dataloader):
        output, acc, loss = trainUnit(makeInput(item, hid), makeTarget(item), model, hp)
        total_acc += acc
        total_count += 1 
        hid = output["hid"]
    return total_acc/total_count, loss





