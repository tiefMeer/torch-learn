
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
    gp.modelSetting(gp.model.parameters())
    return gp.model

class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(gp.vocab_size, gp.embed_dim)
        self.encoder = EncodeModel()
        self.decoder = AttnDecodeModel()
    def greatorMask(self, current_n, lengths):
        mask = current_n < lengths
        return mask.to(gp.device)
    def wipeEOS(self, output, now_len, length):
        for i in range(output.size(0)):
            if now_len < length[i]:
                output[i,0,gp.vocab["<EOS>"]] = 0
                output[i,0,gp.vocab["图"]] = 0
                output[i,0,gp.vocab["折"]] = 0
        return output
    def forward(self, inputs, target):
        loss = 0.0
        pred = torch.zeros(gp.BATCH_SIZE, 0, dtype=int, device=gp.device) # 第2维是序列长
        # encode_output: <batch_size x vocab_size>
        # encode_hidden: <1 x batch_size x hidden_dim>
        inputs[0] = self.embedding(inputs[0])
        encoder_output, encoder_hidden = self.encoder(inputs)
        seq_start = torch.fill_(torch.zeros(gp.BATCH_SIZE, 1), 
                                gp.vocab["<SOS>"]).to(gp.device).int()
        decoder_inputs = [seq_start, encoder_hidden, encoder_output]
        for di in range(max(target[1])):
            decoder_inputs[0] = self.embedding(decoder_inputs[0])
            decoder_output, decoder_hidden = self.decoder(decoder_inputs)
            decoder_output = self.wipeEOS(decoder_output, di, target[1])
            topv, topi = decoder_output.topk(1)
            pred = torch.cat((pred, topi.view(-1, 1)), dim=1)
            loss_tensor = gp.criterion(decoder_output.squeeze(1), 
                                       target[0][:, di]) 
            loss += loss_tensor.masked_select(self.greatorMask(di, target[1])).mean()
            decoder_inputs[0] = target[0][:, di].view(gp.BATCH_SIZE, 1) # Teacher forcing
        return pred, loss
            # :draft:V
            #print("target: ", target[0][:,di].shape,"\n", target[0][:,di])
            #print("predict: ", topi[0][:10], "\tprob: ", decoder_output[0,0,3])
            #print("loss: ", loss_tensor.data.tolist()[:10])
            #print("loss mean: ", loss_tensor.masked_select(self.greatorMask(di, target[1])))
            #print("loss:\n", loss.masked_select(self.greatorMask(di, target[1])))
            # 以下是 non-Teacher forcing part
            #decoder_inputs = topi.squeeze().detach()  # detach from history as input
            #loss += criterion(decoder_output, target[di])
            #if decoder_inputs.item() == EOS_token:
            #    break

class EncodeModel(nn.Module):
    def __init__(self):
        super(EncodeModel, self).__init__()
        # layers
        self.gru = nn.GRU(gp.embed_dim, gp.hidden_dim, batch_first=True)
        self.lsm = nn.LogSoftmax(dim=1)
    def forward(self, inputs):
        embed_text, length = inputs # embed_text: <batch_size x seq_len x embed_dim> 
        gru_in = pack_padded_sequence(embed_text, length, batch_first=True) 
        gru_out, hidden = self.gru(gru_in)
        unpack = pad_packed_sequence(gru_out, batch_first=True)
        output = self.lsm(unpack[0])
        return output, hidden

class AttnDecodeModel(nn.Module):
    def __init__(self, **para):
        super(AttnDecodeModel, self).__init__()
        #para
        self.embed_dim = gp.embed_dim
        self.hidden_dim = gp.hidden_dim
        self.output_size = gp.vocab_size
        self.max_length = gp.max_length
        self.text_max_length = gp.text_max_length
        self.dropout_p = 0.05
        # layers
        self.dropout = nn.Dropout(self.dropout_p)
        self.sm = nn.Softmax(dim=2)
        self.attn = nn.Linear(self.embed_dim + self.hidden_dim,
                              1+self.text_max_length)
        self.attn_combine = nn.Linear(self.embed_dim + self.hidden_dim,
                                      self.hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.output_size)
    def forward(self, inputs):
        embed_prev, hidden, encoder_outputs = inputs    #bx1,1xbx50,bx501x50
        embed = self.dropout(embed_prev)                #bx1x128
        hidden = hidden.permute(1,0,2)           # 为了union
        union = torch.cat((embed, hidden), 2)           #bx1x178
        hidden = hidden.permute(1,0,2)           # 换回来
        attn_w = self.sm(self.attn(union))              #bx1x501
        #print("attn_w[0]: \n", attn_w[0])
        attn_info = torch.bmm(attn_w, encoder_outputs)  #bx1x50
        output = torch.cat((embed, attn_info), 2)       #bx1x178
        output = self.relu(self.attn_combine(output))   #bx1x50
        output, hidden = self.gru(output, hidden)       #bx1x50
        output = self.out(output)                       #bx1xvocab_size
        return output, hidden


#gp = Model(["EncodeModel", "AttnDecodeModel"])

