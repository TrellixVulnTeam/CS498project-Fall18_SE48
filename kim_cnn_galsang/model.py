import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(num_embeddings = self.VOCAB_SIZE + 2,
                                        embedding_dim = self.WORD_DIM,
                                        padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX)) #set embedding weight
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(num_embeddings = self.VOCAB_SIZE + 2,
                                                embedding_dim = self.WORD_DIM,
                                                padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2 # change in_channel to 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(in_channels=self.IN_CHANNEL,
                             out_channels=self.FILTER_NUM[i],
                             kernel_size=self.WORD_DIM * self.FILTERS[i],
                             stride=self.WORD_DIM)

            setattr(self, 'conv_{}'.format(i), conv)

        # self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE) #still confuse the i/o channel

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, input):
        x = self.embedding(input).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(input).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1) # concat the two channel

        conv_results = [
            F.max_pool1d(
                F.relu(self.get_conv(i)(x)),
                       self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])
                                                            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1) # concat conv result from different conv-layer to form a conv result
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x
