
import re
import os
import jieba
import torch
from torchtext.vocab import build_vocab_from_iterator

# Hyper parameters for model and other genernal setting 
# instantiate before runing the model and after defining model
class GlobalParameters():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = {}
        self.EPOCHS = 2
        self.BATCH_SIZE = 4
        self.defaultLR = 0.01
        self.embed_dim = 128
        self.hidden_dim = 30
        self.target_size = 3
        self.dataSourceFilePath = "data/"
        self.vocab_path = "data/vocab.dat"
        self.model_path = "data/model/model.dat"
        self.encode_model_path = "data/model/encode_model.dat"
        self.decode_model_path = "data/model/decode_model.dat"
        self.criterion = torch.nn.CrossEntropyLoss()
        jieba.load_userdict("data/jiebaDict.txt")
    def addVocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    #下面这些改放到model类里
    #def modelSetting(self, model_para):
    #    self.optimizer = torch.optim.SGD(model_para, lr=self.LR)
    #    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

gp = GlobalParameters()

def buildVocab(getIter=None):
    if os.path.exists(gp.vocab_path):
        vocab = torch.load(gp.vocab_path)
    else:
        it = getIter()
        vocab = build_vocab_from_iterator(yield_tokens(it),
                                          specials=["<pad>", "<unk>", "<SOS>", "EOS"])
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, gp.vocab_path)
    gp.addVocab(vocab)

# 将一段字符串文本t分词数字化
def text_pipeline(t):
    return torch.tensor(gp.vocab(list(pretreat(t)))).to(gp.device)

def label_pipeline(x):
    if x=='0' or x=='1' or x=='-1' :
        return int(x)+1
    else:
        return None

def yield_tokens(data_iter):
    for item in data_iter:
        yield pretreat(item["text"])

# 文本预处理, 将输入一段文本分词后清除异常字符, 返回词iter
def pretreat(text):
    with open("data/badWordList.txt") as f:
        badWordList = f.read().split('\n')
        # 匹配全角感叹号，问号，逗号，句号，汉字，数字
        pattern1 = re.compile("[\uff01\uff1f\uff0c\u3002\u4e00-\u9fff0-9]+")
        pattern2 = re.compile("[0-9a-zA-Z]+")
        for word in jieba.cut(text):
        #for word in list(text):
            if pattern1.fullmatch(word) or pattern2.fullmatch(word):
                if word not in badWordList:
                    yield word

