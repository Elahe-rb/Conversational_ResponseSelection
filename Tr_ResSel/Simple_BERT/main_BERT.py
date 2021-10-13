import time
import argparse
import pickle
import os
from finetuning import NeuralNetwork
from argsparams import *

def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)

if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        #format of train, test and dev
        #dict <class 'dict'>:{
        #'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
        #'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    args.is_training = True
    if args.is_training==True:
        print("start training...")
        train_model(train,dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")


