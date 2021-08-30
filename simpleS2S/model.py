
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

from meta import *

def loadModel(model_path=gp.model_path):
    if os.path.exists(model_path) and \
       os.path.getsize(model_path) != 0:
        gp.model = torch.load(model_path)
    else:
        #last_model = max(glob.glob(model_path+".*"))
        last_model = ""     # 用于取最后一个模型备份．暂时不用
        if os.path.exists(last_model) and \
           os.path.getsize(last_model) != 0:
            gp.model = torch.load(last_model)
        else:
            gp.model = Seq2SeqModel().to(gp.device)  # 不存在现有模型，新建一个
    gp.modelSetting()
    return gp.model

class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.dropout_p = 0.05
        self.embedding = nn.Embedding(gp.vocab_size, gp.embed_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.encoder = EncodeModel()
        self.decoder = DecodeModel()
    def initHidden(self):
        return torch.zeros(1, gp.BATCH_SIZE, gp.hidden_dim, device=gp.device)
    #def wipe(self, output, now_len, length):
    #    for i in range(output.size(0)):
    #        if now_len < length[i]:
    #            output[i,0,gp.vocab["<EOS>"]] = 0
    #            output[i,0,gp.vocab["<pad>"]] = 0
    #            output[i,0,gp.vocab["<unk>"]] = 0
    #    return output
    def forward(self, inputs):
        inputs[0] = self.embedding(inputs[0])
        hidden = self.initHidden()
        for i in range(inputs[1][0]):
            # inputs[0]: <batch_size x vocab_size>
            # hidden: <1 x batch_size x hidden_dim>
            encoder_inputs = [inputs[0][:,i,:].view(1,1,-1), inputs[1], hidden]
            encoder_output, hidden = self.encoder(encoder_inputs)

        pred = torch.zeros(gp.BATCH_SIZE, 0, 
                           dtype=int, device=gp.device) # 第2维是序列长
        pred_prob = torch.zeros(gp.BATCH_SIZE, 0, gp.vocab_size, 
                                dtype=int, device=gp.device) 
        pred = torch.cat((pred, 
                          torch.fill_(torch.zeros(gp.BATCH_SIZE, 1), 
                                      gp.vocab["<SOS>"]).to(gp.device).int()),
                         dim=1)
        #print(hidden)
        #input()

        for di in range(inputs[1][0]):
            decoder_inputs = [pred[:,di].unsqueeze(1), hidden, encoder_output]
            decoder_inputs[0] = self.embedding(decoder_inputs[0])
            decoder_output, hidden = self.decoder(decoder_inputs)
            #decoder_output = self.wipe(decoder_output, di, inputs[1])
            pred_prob = torch.cat((pred_prob, decoder_output), dim=1)
            topv, topi = decoder_output.topk(1)
            pred = torch.cat((pred, topi.view(-1, 1).detach()), dim=1)
        return pred, pred_prob

class EncodeModel(nn.Module):
    def __init__(self):
        super(EncodeModel, self).__init__()
        # layers
        self.relu = nn.ReLU()
        self.gru = nn.GRU(gp.embed_dim, gp.hidden_dim, batch_first=True)
        self.lsm = nn.LogSoftmax(dim=1)
    def forward(self, inputs):
        embed_text, length, hidden = inputs # embed_text: <batch_size x seq_len x embed_dim> 
        embed_text = self.relu(embed_text)
        output, hidden = self.gru(embed_text, hidden)

        #gru_in = pack_padded_sequence(embed_text, length, batch_first=True) 
        #gru_out, hidden = self.gru(gru_in, hidden)
        #unpack = pad_packed_sequence(gru_out, batch_first=True)
        #output = self.lsm(unpack[0])
        return output, hidden

class DecodeModel(nn.Module):
    def __init__(self, **para):
        super(DecodeModel, self).__init__()
        self.gru = nn.RNN(gp.embed_dim, gp.hidden_dim, batch_first=True)
        self.out = nn.Linear(gp.hidden_dim, gp.vocab_size)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        embed_prev, hidden, encoder_outputs = inputs    #bx1,1xbx50,bx501x50
        output, hidden = self.gru(embed_prev, hidden)
        output = self.relu(self.out(output))            #bx1xvocab_size
        return output, hidden


