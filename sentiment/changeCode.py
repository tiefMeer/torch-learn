#!/usr/bin/env python

import codecs

fin = codecs.open("nCoV_900k_train.unlabled.csv",'r', "gb18030")
fout = codecs.open("nCoV_900k_train.unlabled.utf.csv",'w', "utf-8")

#fin = codecs.open("nCoV_100k_train.labled.csv",'r', "gb18030")
#fout = codecs.open("nCoV_100k_train.labled.utf.csv",'w', "utf-8")

num = 1
line = fin.readline()
while line:
    try:
        fout.write(line)
        line = fin.readline()
        num += 1
    except UnicodeDecodeError: 
        print(num, '\t', line)
        line = fin.readline()
        num += 1


