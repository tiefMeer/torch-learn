#!/bin/bash

logfolder0="data/sentiment/log/"
modelfolder0="data/sentiment/model/"

handleLog(){
    cd $1
    cat *.log |sed -n '/acc:/p'|cut  -d'(' -f2 |cut -d',' -f1 |tee acc.dat
    #cat *.log |sed -n '/acc:/p'|cut  -d'(' -f2 | cut -f',' -f1 |tee acc.dat
    cat *.log |sed -n '/accu/p'|cut  -d' ' -f13 | tee accu.dat
    python /home/jovyan/lab/torch-learn/sentiment/draw.py
    cd - 
}

#handleLog "data/sentiment/log/30-02:5708/"

for i in $(seq 1 100)
do
    logfolder="${logfolder0}$(date +%d-%H:%M%S)/"
    mkdir $logfolder
    for i in $(seq 1 50)
    do
        logpath="${logfolder}train.$(date +%d-%H:%M%S).log"
        ./train.py |tee $logpath 
    done
    handleLog $logfolder &
    cp ${modelfolder0}model.dat ${modelfolder0}/bak/model.dat.$(date +%d-%H:%M%S).bak &
done

