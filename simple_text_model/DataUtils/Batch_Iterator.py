# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:55
# @File : Batch_Iterator.py.py
# @Last Modify Time : 2018/1/30 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Batch_Iterator.py
    FUNCTION : None
"""

import torch
from torch.autograd import Variable
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Batch_Features:
    """
    Batch_Features
    """
    def __init__(self):

        self.batch_length = 0
        self.inst = None
        self.word_features = None
        self.label_features = None
        self.sentence_length = []

    @staticmethod
    def cuda(features):
        """
        :param features:
        :return:
        """
        features.word_features = features.word_features.cuda()
        features.label_features = features.label_features.cuda()


class Iterators:
    """
    Iterators
    """
    def __init__(self, batch_size=None, data=None, operator=None, config=None):
        self.config = config
        self.batch_size = batch_size
        self.data = data
        self.operator = operator
        self.operator_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self):
        """
        :param batch_size:  batch size
        :param data:  data
        :param operator:
        :param config:
        :return:
        """
        assert isinstance(self.data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(self.batch_size, list), "ERROR: batch_size must be in list [16,1,1]"

        for id_data in range(len(self.data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self._convert_word2id(self.data[id_data], self.operator)
            self.features = self._Create_Each_Iterator(insts=self.data[id_data],
                                                       batch_size=self.batch_size[id_data],
                                                       operator=self.operator)
            self.data_iter.append(self.features)
            self.features = []

        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def _convert_word2id(insts, operator):
        """
        :param insts:
        :param operator:
        :return:
        """

        for inst in insts:
            # word
            for index in range(inst.words_size):
                word = inst.words[index]
                wordId = operator.word_alphabet.from_string(word)
                if wordId == -1:
                    wordId = operator.word_unkId
                inst.words_index.append(wordId)

            # label
            label = inst.labels[0]
            labelId = operator.label_alphabet.from_string(label)
            inst.label_index.append(labelId)


    def _Create_Each_Iterator(self, insts, batch_size, operator):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1

            # when one batch is full or reach the end
            if len(batch) == batch_size or count_inst == len(insts):
                one_batch = self._Create_Each_Batch(insts=batch, batch_size=batch_size, operator=operator)
                self.features.append(one_batch)
                batch = []

        print("The all data has created iterator.")
        return self.features


    def _Create_Each_Batch(self, insts, batch_size, operator):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """

        # print("create one batch......")
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        sentence_length = []
        for inst in insts:
            sentence_length.append(inst.words_size)
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size


        # create with the Tensor/Variable
        # word features
        batch_word_features = Variable(torch.zeros(batch_length, max_word_size).type(torch.LongTensor))
        # label feature
        batch_label_features = Variable(torch.zeros(batch_length * 1).type(torch.LongTensor))

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features.data[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features.data[id_inst][id_word_index] = operator.word_paddingId

            # label
            batch_label_features.data[id_inst] = inst.label_index[0]
            # batch_label_features.data[id_inst] = inst.label_index[-1]

        # batch
        features = Batch_Features()
        features.inst = insts
        features.word_features = batch_word_features
        features.sentence_length = sentence_length
        features.batch_length = batch_length
        features.label_features = batch_label_features

        # features.word_features = sorted_inputs_words
        # features.sentence_length = sorted_seq_lengths
        # features.desorted_indices = desorted_indices
        # features.desorted_indices = None

        if self.config.use_cuda is True:
            features.cuda(features)
        return features




