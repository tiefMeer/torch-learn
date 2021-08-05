#!/usr/bin/env python

import os 
import glob
import matplotlib.pyplot as plt

from collections import Counter

from train import *

# 对给定list-like输入给出计数dict并画图
def countPlot(li, name="count.png", prefix="data/fig/count/"):
    count=Counter(li)
    plt.cla()
    plt.plot([count[i] for i in range(max(li))])
    plt.savefig(prefix+name)
    return count

def test():
    buildVocab(lambda: getDataIter(fields[0]))
    dataloader = getDataloader()
    t = next(iter(dataloader))[0][0]
    t = next(iter(dataloader))
    t0 = t[0][0].data.tolist()
    t1 = t[2][0].data.tolist()
    print(digital2text(t0))
    print(digital2text(t1))
    print(t[0].shape)
    print(t[2].shape)

if __name__ == "__main__":
    test()

