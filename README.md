# CS498 Deep Learning Project

## C-LSTM Implementation
> Paper: [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)  
> Code Reference: [GitHub: cnn-lstm-bilstm-deepcnn-clstm-in-pytorch](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch)  
> Contributor: Siwei  
> Last Update: 2018-11-14  

### Current Data
SST-1 and SST-2 from original [GitHub Repository](https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/Data).

### Current Configuration
```conf
[Model]
LSTM = True

[Optimizer]
learning_rate = 0.001
SGD = True
optim_momentum_value = 0.9

[Train]
num_threads = 1
device = 0
cuda = True
epochs = 10
batch_size = 16
log_interval = 5
test_interval = 100
save_interval = 100
```

### Current Test Accuracy
> Location: [zhou_lstm/result/lstm_sgd.log](zhou_lstm/result/lstm_sgd.log)
```
Dev  Accuracy:  Evaluation - loss: 0.655496  acc: 62.3853%(544/872) 
Test Accuracy:  Evaluation - loss: 0.660529  acc: 61.3399%(1117/1821) 
The Current Best Dev Accuracy: 64.7936, and Test Accuracy is :62.8226, locate on 10 epoch.
```
