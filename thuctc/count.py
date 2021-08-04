#!/usr/bin/env python

import os 
import glob
import matplotlib.pyplot as plt

from collections import Counter

types=os.listdir("data/THUCNews/")
for t in types:
    files=glob.glob("data/THUCNews/"+t+"/*.txt")
    sizes=[os.path.getsize(f) for f in files]
    m=max(sizes)
    count=Counter(sizes)
    plt.cla()
    plt.plot([count[i] for i in range(m)])
    plt.savefig("data/fig/"t+"count.png")

