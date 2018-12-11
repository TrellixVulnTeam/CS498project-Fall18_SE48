# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main_hyperparams.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main_hyperparams.py.py
    FUNCTION : main
"""

#! /usr/bin/env python
import argparse
import datetime
import Config.config as configurable
from DataUtils.mainHelp import *
from DataUtils.Alphabet import *
from test import load_test_data
from test import T_Inference
from trainer import Train
import random

# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)


def start_train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param config:  config
    :return:  None
    """
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()
    print("Finish Train.")


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param alphabet:  alphabet dict
    :param config:  config
    :return:  None
    """
    print("\nTesting Start......")
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = T_Inference(model=model, data=data, path_source = path_source, path_result= path_result,
                         alphabet=alphabet, use_crf=config.use_crf, config=config)
    infer.infer2file()
    print("Finished Test.")


def sentence_demo(train_iter, dev_iter, test_iter, model, alphabet, config, sentence):
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = T_Inference(model=model, data=data, path_source = path_source, path_result= path_result,
                         alphabet=alphabet, use_crf=config.use_crf, config=config)
    infer.oneSentenceInf(sentence)
    print("Finished Test.")


def main():
    """
    main()
    :return:
    """

    # 1.save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    # 2.get data, iter, alphabet, load_data is in mainHelp.py
    sentence = "The moive is good!"
    train_iter, dev_iter, test_iter, alphabet = load_data(config=config, sentence=sentence)

    # 3.get params
    get_params(config=config, alphabet=alphabet)

    # 4.save dictionary
    save_dictionary(config=config)

    model = load_model(config)

    # 5.print("Training Start......")
    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.test is True:
        start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
        exit()
    elif config.demo is True:
        sentence_demo(train_iter, dev_iter, test_iter, model, alphabet, config, sentence)
        exit()


def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config",
                        dest="config_file", type=str, default="./Config/config.cfg",help="config path")
    parser.add_argument("--train",
                        dest="train", type=bool, default=False, help="train model")
    parser.add_argument("--demo",
                        dest="demo", type=bool, default=True, help="predict for one sentence")
    parser.add_argument("-t", "--test",
                        dest="test", type=bool, default=False, help="test model")
    parser.add_argument("-p", "--process",
                        dest="process", type=bool, default=True, help="data process")
    parser.add_argument("--t_model",
                        dest="t_model", type=str, default="./Save_BModel/text_model.pt", help="model for test")
    parser.add_argument("--t_data",
                        dest="t_data", type=str, default="test", help="data[train dev test None] for test model")
    parser.add_argument("--data_name",
                        dest="d_name", type=str, default="sst", help="the name of dataset")
    parser.add_argument("--predict",
                        dest="predict", type=bool, default=False, help="predict model")
    parser.add_argument("--use_crf",
                        dest="use_crf", type=bool, default=False, help="whether use crf in evaluation")
    parser.add_argument("--model",
                        dest="model", type=str, default="cnn_bilstm_att", choices=["cnn", "bilstm", "cnn_bilstm_att"])
    parser.add_argument("--dict_directory",
                        dest="dict_directory", type=str, default="./Save_dictionary_cnn", help="dictionary save path")
    parser.add_argument("--save_direction",
                        dest="save_direction", type=str, default="./save_direction_cnn", help="model check point save path")

    parser.add_argument("--save_best_model_dir",
                        dest="save_best_model_dir", type=str, default="./Save_BModel_CNN",
                        help="best model directory")
    parser.add_argument("--model_name",
                        dest="model_name", type=str, default="cnn_model", help="model name")


    args = parser.parse_args()

    # 1.load config_file into config
    config = configurable.Configurable(config_file=args.config_file)
    # 2.add additional arguments into config
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.demo = args.demo
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    config.d_name = args.d_name
    config.use_crf = args.use_crf
    config.model = args.model
    config.dict_directory = args.dict_directory
    config.model_name = args.model_name

    if config.model == "cnn":
        config.save_best_model_dir = "./Save_BModel_CNN"
        config.dict_directory = "./Save_dictionary_cnn"
        name = "cnn_model_" + config.d_name
        config.t_model = "./Save_BModel_CNN/"+name+".pt"
        config.model_name = name

    elif config.model == "bilstm":
        config.save_best_model_dir = "./Save_BModel_BiLSTM"
        config.dict_directory = "./Save_dictionary_bilstm"
        name = "bilstm_model_" + config.d_name
        config.t_model = "./Save_BModel_BiLSTM/"+name+".pt"
        config.model_name = name
    elif config.model == "cnn_bilstm_att":
        config.save_best_model_dir = "./Save_BModel_C-BiLSTM-Attention"
        config.dict_directory = "./Save_dictionary_cnn_bilstm_att"
        name = "cnn_bilstm_attention_model_" + config.d_name
        config.t_model = "./Save_BModel_C-BiLSTM-Attention/"+name+".pt"
        config.model_name = name


    # 3.check config
    if config.test is True or config.demo is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        exit()

    # 4.print configuration
    print("***************************************")
    print("Data Process : {}".format(config.process))
    print("Train model : {}".format(config.train))
    print("Test model : {}".format(config.test))
    print("demo model : {}".format(config.demo))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("***************************************")

    return config


if __name__ == "__main__":

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    # get arguments
    config = parse_argument()
    # set seed for GPU, if GPU is available
    if config.use_cuda is True:
        print("Using GPU To Train......")
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())

    main()

