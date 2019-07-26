import torch.nn as nn
import torch
import numpy as np

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        # Added to fix error "'indices' to have scalar type Long" when calling with built-in inference function.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reviews = torch.tensor(reviews.tolist(), dtype=torch.long, device=device)
        #
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        # Change to fix error "indices must be long" when calling with built-in inference function.
        normalized_lengths = np.asarray([min(length, 499) for length in lengths.cpu()], dtype=np.int64)
        last_word_idx = normalized_lengths - 1
        out = out[last_word_idx, np.asarray(range(len(lengths)), dtype=np.int64)] # select the last y_hat from each sequence
        # out = out[lengths - 1, range(len(lengths))]
        #
        return self.sig(out.squeeze())