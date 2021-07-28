#!/usr/bin/env python3

from train import *

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

