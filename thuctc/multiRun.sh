#!/bin/bash

logfolder0="data/log/"
figfolder0="data/fig/"
modelfolder0="data/model/"


for i in $(seq 1 100)
do
    logpath="${logfolder0}train.$(date +%d-%H:%M%S).log"
    python ./train.py |tee $logpath 
    cp ${figfolder0}epoch_loss.png  ${figfolder0}epoch_loss-${i}.png
    #cp ${modelfolder0}model.dat ${modelfolder0}/bak/model.dat.$(date +%d-%H:%M%S).bak &
done

