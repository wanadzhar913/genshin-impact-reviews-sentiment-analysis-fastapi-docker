import torch
from torch import nn

class SentimentRNN(nn.Module):
    def __init__(self,
            no_layers,
            vocab_size,
            output_dim,
            hidden_dim,
            embedding_dim,
            drop_prob=0.5,
            device="cpu"
        ):
        super(SentimentRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)

        # initialize hidden state
        hidden = self.init_hidden(batch_size)

        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True

        # print(embeds.shape)  # e.g., [50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        # get last batch of labels
        sig_out = sig_out[:, -1]

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        '''
        Initializes hidden state: https://discuss.pytorch.org/t/how-to-handle-last-batch-in-lstm-hidden-state/40858
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device)
        
        hidden = (h0,c0)

        return hidden