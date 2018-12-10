# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : CNN_BiLSTM_Attention.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)
"""
    Neural Network: CNN_BiLSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""


class CNN_BiLSTM_Attention(nn.Module):

    def __init__(self, **kwargs):
        super(CNN_BiLSTM_Attention, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        # self.args = args
        self.hidden_dim = self.lstm_hidden_dim
        self.num_layers = self.lstm_num_layers
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        self.C = C
        Ci = 1
        Co = self.kernel_num
        Ks = self.kernel_sizes

        self.embed = nn.Embedding(V, D, padding_idx=self.paddingId)
        # pretrained  embedding
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks]
        print(self.convs1)
        # for cnn cuda
        if self.use_cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True, bias=True)
        #attention
        self.attn = nn.Linear(self.hidden_dim,1)
        # linear
        L = len(Ks) * Co + self.hidden_dim #* 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, sentence_length):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        # cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        H = bilstm_out[:,:,:self.hidden_dim] + bilstm_out[:,:,self.hidden_dim:]
        M = F.tanh(H)
        alpha = F.softmax(self.attn(M))
        # alpha = torch.transpose(alpha,1,0)
        # alpha = torch.transpose(alpha,2,1)
        H = torch.transpose(H,2,1)
        r = torch.bmm(H,alpha)
        r = r.squeeze(2)
        h_star = F.tanh(r)
        h_star = self.dropout(h_star)
        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        # bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

        # bilstm_out = F.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        # cnn_x = torch.transpose(cnn_x, 0, 1)
        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, h_star), 1)
        # cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit
