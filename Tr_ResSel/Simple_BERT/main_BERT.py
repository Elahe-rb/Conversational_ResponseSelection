import time
import argparse
import pickle
import os
from finetuning import NeuralNetwork
from argsparams import *
import pickletools

def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def eval_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)

if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task]+"train_1M.pkl", 'rb') as f:
        #format of train, test and dev
        #dict <class 'dict'>:{
        #'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
        #'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        print("loading train data...")
        train = pickle.load(f, encoding='ISO-8859-1')
    with open(FT_data[args.task]+"valid_1M.pkl", 'rb') as f:
        #format of train, test and dev
        #dict <class 'dict'>:{
        #'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
        #'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        print("loading valid data...")
        dev = pickle.load(f, encoding='ISO-8859-1')

    args.is_training = True
    if args.is_training==True:
        print("start training...")
        train_model(train,dev)
        with open(FT_data[args.task] + "test_1M.pkl", 'rb') as f:
            # format of train, test and dev
            # dict <class 'dict'>:{
            # 'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
            # 'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
            print("loading test data...")
            test = pickle.load(f, encoding='ISO-8859-1')

        eval_model(test)
    else:
        with open(FT_data[args.task] + "test_1M.pkl", 'rb') as f:
            # format of train, test and dev
            # dict <class 'dict'>:{
            # 'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
            # 'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
            print("loading test data...")
            test = pickle.load(f, encoding='ISO-8859-1')
        eval_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")


