from __future__ import print_function, division
import os
import sys
import time
import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import data_helpers

class CNN(nn.Module):
    def __init__(self, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300,
                    pretrained_embeddings=None, vocab_size=None, sentence_len=None,
                                    mode=None, ConvMethod=None, num_classes=None):

        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = (mode=="nonstatic")
        self.ConvMethod = ConvMethod

        use_cuda = torch.cuda.is_available()

        if use_cuda:
            self.embedding = self.embedding.cuda()

        conv_blocks = [] #blocks list
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1

            if ConvMethod == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters,
                                   kernel_size = kernel_size, stride = 1)
            else:
                conv1d = nn.Conv1d(in_channels = 1, out_channels = num_filters,
                                   kernel_size = kernel_size*embedding_dim, stride = embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            )

            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)

        # ModuleList is needed for registering parameters in conv_blocks
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)

        if self.ConvMethod == "in_channel__is_embedding_dim":
            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            # needs to convert x to (batch, embedding_dim, sentence_len)
            x = x.transpose(1,2)
        else:
            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            # needs to convert x to (batch, 1, sentence_len*embedding_dim)
            x = x.view(x.size(0), 1, -1)

        # x_list is conv results from different conv layers
        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2) #??
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        result = F.softmax(self.fc(out), dim=1)
        return result, feature_extracted


def evaluate(model, x_test, y_test):
    inputs = Variable(x_test)
    preds, vector = model(inputs) # model has been set to predict
    preds = torch.max(preds, 1)[1]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        preds = preds.cuda()

    eval_acc = (preds.data == y_test).sum().item() / len(y_test) # pytorch 0.4
    return eval_acc, vector.cpu().data.numpy()


def load_pretrained_embeddings():
    pretrained_fpath_saved = os.path.expanduser("models/googlenews_extracted-python{}.pl".format(sys.version_info.major))
    if os.path.exists(pretrained_fpath_saved):
        with open(pretrained_fpath_saved, 'rb') as f:
            embedding_weights = pickle.load(f)
    else:
        print('- Error: file not found : {}\n'.format(pretrained_fpath_saved))
        print('- Please run the code "python utils.py" to generate the file first\n\n')
        sys.exit()


    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}
    # added list() to convert dict_values to a list for use in python 3
    out = np.array(list(embedding_weights.values()))
    #np.random.shuffle(out)

    print('embedding_weights shape:', out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out



def train_test_one_split(X, Y, cv, train_index, test_index,
                         batch_size, kernel_sizes, num_filters, embedding_dim,
                         pretrained_embeddings, vocab_size, sentence_len, mode, ConvMethod, num_classes):

    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()

    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)


    use_cuda = torch.cuda.is_available() #check if cuda available
    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    # vocab_size, sentence_len, mode, ConvMethod
    model = CNN(kernel_sizes, num_filters, embedding_dim,
                pretrained_embeddings, vocab_size, sentence_len, mode, ConvMethod, num_classes)

    if cv==0:
        print("\n{}\n".format(str(model)))

    if use_cuda:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0002)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        tic = time.time()

        ########--1.train--###########
        # 1.1 set model to train
        model.train()
        # 1.2 batch SGD
        for i, (inputs, labels) in enumerate(train_loader):
            # NOTE: PyTorch 0.4 merges the Variable and Tensor class into one
            # 1.2.1 batch forward
            if use_cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                preds, _ = model(inputs)
                preds = preds.cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                preds, _ = model(inputs)
            ## 1.2.2 batch backward
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ########--2.evaluate--###########
        model.eval()
        eval_acc, sentence_vector = evaluate(model, x_test, y_test)

        ########--3.summary--############
        print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'
                                .format(epoch, loss.item(), eval_acc, time.time()-tic))

    return eval_acc, sentence_vector




def do_cnn(X, Y, batch_size, kernel_sizes, num_filters, embedding_dim, pretrained_embeddings,
                                                    vocab_size, sentence_len, mode, ConvMethod, num_classes):
    cv_folds = 10
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    acc_list = []
    tic = time.time()
    sentence_vectors, y_tests = [], []
    for cv, (train_index, test_index) in enumerate(kf.split(X)): #get index from the kFold's enumrate
        acc, sentence_vec = train_test_one_split(X, Y, cv, train_index, test_index,
                                                 batch_size,  kernel_sizes, num_filters,
                                                 embedding_dim, pretrained_embeddings,
                                                 vocab_size, sentence_len, mode, ConvMethod, num_classes)

        print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
        acc_list.append(acc)
        sentence_vectors += sentence_vec.tolist()
        y_tests += Y[test_index].tolist()

    print('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format(sum(acc_list)/len(acc_list), time.time()-tic))

    # save extracted sentence vectors in case that we can reuse it for other purpose (e.g. used as input to an SVM classifier)
    # each vector can be used as a fixed-length dense vector representation of a sentence
    np.save('models/sentence_vectors.npy', np.array(sentence_vectors))
    np.save('models/sentence_vectors_y.npy', np.array(y_tests))



def main():
    # for obtaining reproducible results
    np.random.seed(0)
    torch.manual_seed(0)

    embedding_dim = 300
    num_filters = 100
    kernel_sizes = [3, 4, 5]
    batch_size = 50

    use_cuda = torch.cuda.is_available()
    mode = "static"     # mode = "nonstatic"
    use_pretrained_embeddings = True    # use_pretrained_embeddings = False
    ConvMethod = "in_channel__is_1"  # ConvMethod = "in_channel__is_embedding_dim"

    X, Y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocab_size = len(vocabulary_inv_list)
    sentence_len = X.shape[1]
    num_classes = int(max(Y)) + 1  # added int() to convert np.int64 to int

    print('use_cuda = {}\n'.format(use_cuda))
    print('MODE      = {}'.format(mode))
    print('EMBEDDING = {}\n'.format("pretrained" if use_pretrained_embeddings else "random"))
    print('vocab size       = {}'.format(vocab_size))
    print('max sentence len = {}'.format(sentence_len))
    print('num of classes   = {}'.format(num_classes))

    if use_pretrained_embeddings:
        pretrained_embeddings = load_pretrained_embeddings()
    else:
        pretrained_embeddings = np.random.uniform(-0.01, -0.01, size=(vocab_size, embedding_dim))


    do_cnn(X, Y, batch_size, kernel_sizes, num_filters,
           embedding_dim, pretrained_embeddings, vocab_size,
           sentence_len, mode, ConvMethod ,num_classes)

if __name__ == "__main__":
    main()
