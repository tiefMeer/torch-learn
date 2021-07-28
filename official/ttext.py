import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
weight_matrix = TEXT.vocab.vectors

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2 
        return y

def main():
	model = LSTM()
	model.train()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
	loss_funtion = F.cross_entropy
	
	for epoch, batch in enumerate(train_iter):
	    optimizer.zero_grad()
	    start = time.time()
	    # text = batch.text.permute(1, 0)
	    predicted = model(batch.comment_text)
	
	    loss = loss_funtion(predicted, batch.toxic)
	    loss.backward()
	    optimizer.step()
	    print(loss)
	    
if __name__ == '__main__':
    main()	
