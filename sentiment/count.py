#!/usr/bin/env python

import csv 
import jieba
import matplotlib.pyplot as plt

#import importlib
#from ..template.utils import *

from meta import *

from torchtext.data.datasets_utils import _RawTextIterableDataset

def getDataIter(filepath = gp.dataSourceFilePath):
    with open(filepath) as f:
        rd = csv.reader(f)
        if "100" in filepath:
            print("using 100 train dataset.")
            for i in rd:
                t = text_pipeline(i[3])
                l = label_pipeline(i[6])
                if len(t) != 0 and l!=None:
                    yield {"text": t, "senti": l}
        elif "900" in filepath:
            for i in rd:
                t = text_pipeline(i[3])
                if len(t) != 0:
                    yield {"text": t}
        elif "test" in filepath:
            print("using test dataset.")
            for i in rd:
                t = text_pipeline(i[3])
                if len(t) == 0:
                    print(int(i[0]), ",", 0)
                else:
                    yield {"id":int(i[0]), "text": t}
        else:
            raise FileError(filepath)

def count():
    count = [0 for i in range(300)]
    for text in map(lambda x: x["text"], getDataIter(filepath)):
        l=len(text)
        try:
            count[l] += 1
        except:
            print(l)
            print(text)

    print(count)
    plt.plot(count)
    plt.show()


def labledDataset():
    NAME = "100k-labeld"
    NUM_ITEMS = 99999
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter())

def testDataset():
    NAME = "test"
    NUM_ITEMS = 10000
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter(gp.testDataSourceFilePath))
