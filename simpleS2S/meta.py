
import re
import os
import jieba
import torch
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

# Hyper parameters for model and other genernal setting 
# instantiate before runing the model and after defining model
class GlobalParameters():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = {}
        self.EPOCHS = 500
        self.BATCH_SIZE = 1
        self.LR = 0.002
        self.embed_dim = 128
        self.hidden_dim = 128
        self.target_size = 3
        self.text_select_width = 10
        self.max_length = 30
        self.text_max_length = 400
        self.dataSourceFilePath = "data/"
        self.vocab_path = "data/vocab.dat"
        self.fig_path = "data/fig/"
        self.epoch_loss_list = []
        self.epoch_loss_list_fig_path = "data/fig/epoch_loss.png"
        self.model_path = "data/model/model.dat"
        self.encode_model_path = "data/model/encode_model.dat"
        self.decode_model_path = "data/model/decode_model.dat"
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        jieba.load_userdict("data/jiebaDict.txt")
        with open("data/badWordList.txt") as f:
            self.bad_word_list = f.read().split('\n')
    def addVocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    def modelSetting(self, model_para=None):
        model_para = model_para or self.model.parameters()  
        self.optimizer = torch.optim.SGD(model_para, lr=self.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

gp = GlobalParameters()

def buildVocab(getIter=None):
    if os.path.exists(gp.vocab_path):
        vocab = torch.load(gp.vocab_path)
    else:
        it = getIter()
        vocab = build_vocab_from_iterator(yield_tokens(it),
                                          specials=["<pad>", "<unk>", "<SOS>", "<EOS>"])
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, gp.vocab_path)
    gp.addVocab(vocab)

# ???std????????????????????????t???????????????
# ?????????????????????????????????
def textSlice(t, std, res=[]):
    L, W = gp.text_max_length, gp.text_select_width
    i = max(enumerate(std), key=lambda x:x[1][1])[0]
    try:
        anchor= t.index(std[i][0])
    except ValueError: # ???????????????i not in list t,??????i????????????-W+W ????????????
        std.pop(i)
        new_t = t
    else:
        std[i] = std[i][0], std[i][1]-1
        if std[i][1] == 0:
            std.pop(i)
        res.extend(t[anchor-W: anchor+W])
        new_t = t[:max(0, anchor-W)] + t[anchor+W:]

    if len(t)<W or len(res)>L or len(std)==0:
        return res[:L]
    else:
        return textSlice(new_t, std, res=res)

# ??????????????????????????????????????????,?????????????????????
def textTidy(t):
    L = gp.text_max_length
    if len(t) <= L:
        return t
    # ???t????????????20??????maxn: [(???idx, t???????????????),]
    maxn = sorted(Counter(t).items(), key=lambda x:x[1], reverse=True)[:20]
    res = textSlice(t, maxn, res=[])
    return res

# ????????????????????????t???????????????
def text_pipeline(t):
    return torch.tensor(textTidy(gp.vocab(list(pretreat(t))))).to(gp.device)

def title_pipeline(t):
    return torch.tensor(gp.vocab(list(pretreat(t)))).to(gp.device)

def label_pipeline(x):
    if x=='0' or x=='1' or x=='-1' :
        return int(x)+1
    else:
        return None

def yield_tokens(data_iter):
    for item in data_iter:
        yield pretreat(item)

# ???????????????, ????????????????????????????????????????????????, ?????????iter
def pretreat(text, use_bad_word_list=True):
    # bad??????????????????????????????????????????????????????????????????
    #pattern1 = re.compile("[\uff01\uff1f\uff0c\u3002\u4e00-\u9fff0-9]+")
    # ?????????????????????
    pattern1 = re.compile("[\u4e00-\u9fff0-9]+")
    pattern2 = re.compile("[0-9a-zA-Z]+")
    for word in jieba.cut(text):
    #for word in list(text):
        if pattern1.fullmatch(word) or pattern2.fullmatch(word):
            if use_bad_word_list:
                if word not in gp.bad_word_list:
                    yield word
            else:
                yield word






