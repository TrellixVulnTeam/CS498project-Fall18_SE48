# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import sys
import os
import re
import random
import torch
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)

# clean data
class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config, dataset, demo):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        self.dataset = dataset
        self.demo = demo

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        dataset = self.dataset
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))

        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_Data(path=path[id_data], shuffle=shuffle, dataset = dataset, demo = self.demo)

            if shuffle is True and id_data == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            # sorted(inst)
            # if id_data == 0:
            #     insts = self._sort(insts)
            # sorted_insts = self.sort(insts)
            self.data_list.append(insts)

        # return train/dev/test data
        # path=[config.train_file, config.dev_file, config.test_file]
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]


    def _Load_Each_Data(self, path=None, shuffle=False, dataset=None, demo=False):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0



        # 1.open the file
        with open(path, encoding="UTF-8") as f:

            # 2.Load data into instance format
            inst = Instance()

            # 3.Load each line
            for line in f.readlines():

                # 4.strip leading whitespace are removed
                line = line.strip()
                now_lines += 1
                if now_lines % 200 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                if line == "\n":
                    print("empty line")


                inst = Instance()

                if demo and path == "./Data/demo_sentence.txt":
                    word = line
                    label = '?'
                elif dataset == 'trec':
                    line = line.split(':')
                    label = line[0]
                    word = line[1]
                    if label not in ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]:
                        print("Error line: ", " ".join(line))
                        continue
                else:
                    line = line.split()
                    label = None
                    word = None

                    if dataset == 'sst':
                        label = line[0]
                        word = " ".join(line[1:])
                        if label not in ["0", "1"]:
                            print("Error line: ", " ".join(line))
                            continue
                    elif dataset == 'mr':
                        label = line[-1]
                        word = " ".join(line[:-6])
                        if label not in ["0", "1", "2", "3", "4"]:
                            print("Error line: ", " ".join(line))
                            continue


                inst.words = self._clean_str(word).split()
                inst.labels.append(label)
                inst.words_size = len(inst.words)
                insts.append(inst)

                if len(insts) == self.max_count:
                    break
            # print("\n")
        return insts



