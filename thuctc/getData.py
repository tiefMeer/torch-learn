#!/usr/bin/env python

import glob

fields = glob.glob("data/*")

def getDataIter(field):
    res={}
    for item in glob.iglob(field+"/*"):
        with open(item) as f:
            res["label"] = f.readline().strip()
            res["text"]  = f.read().strip()
        yield res

def getDataset(field):
    NAME = field
    NUM_ITEMS = len(glob.glob(field)) 
    return _RawTextIterableDataset(NAME, NUM_ITEMS, getDataIter())
    
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

