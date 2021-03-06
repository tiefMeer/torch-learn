#!/usr/bin/env python

import time
import matplotlib.pyplot as plt

fig_folder="/home/jovyan/lab/torch-learn/sentiment/data/sentiment/fig/"

def draw():
    acc=list(map(int, open("acc.dat").read().strip().split()))
    count=list(map(lambda x:acc.count(x),range(1, max(acc)+1)))
    plt.cla()
    plt.plot(count)
    plt.savefig(fig_folder+str(time.time_ns())+".count_acc.png")
    accu=list(map(float, open("accu.dat").read().strip().split()))
    plt.cla()
    plt.plot(accu)
    plt.savefig(fig_folder+str(time.time_ns())+".accu.png")



draw()


