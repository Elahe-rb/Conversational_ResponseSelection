import time
import argparse
import numpy as np
import pickle
import os
from finetuning import NeuralNetwork
from argsparams import *
import pickle

def train_model(train, dev, arg_params):
    model = NeuralNetwork(args=arg_params)
    model.fit(train, dev)

def eval_model(test, arg_params):
    model = NeuralNetwork(args=arg_params)
    model.load_model(arg_params.savePath+str(arg_params.task)+"_"+str(arg_params.network_num)+".pt")
    model.evaluate(test, is_test=True)

def run(arg_params):
    start = time.time()
    with open(arg_params.dataPath + "dataset_" + str(arg_params.network_num) + ".pkl", 'rb') as f:
        print("loading data...")
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if arg_params.is_training == True:
        print("start training...")
        train_model(train, dev)
        print("test model...")
        eval_model(test)
    else:
        print("test model...")
        eval_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")


if __name__ == '__main__':
    run(args)


'''
if __name__ == '__main__':
    start = time.time()
    with open(args.dataPath+"ubuntu_data/test_1M.pkl", 'rb') as f:
        #format of train, test and dev
        #dict <class 'dict'>:{
        #'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
        #'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        print("loading train data...")
    #train = np.load(FT_data[args.task]+'dev_1M.pkl')
        train = pickle.load(f, encoding='ISO-8859-1')
    #with open(FT_data[args.task]+"valid_1M.pkl", 'rb') as f:
        #format of train, test and dev
        #dict <class 'dict'>:{
        #'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
        #'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        #print("loading valid data...")
    #dev = np.load(FT_data[args.task] + 'dev_1M_optimized.npy', allow_pickle=True)
        #dev = pickle.load(f, encoding='ISO-8859-1')

    args.is_training = True
    if args.is_training==True:
        print("start training...")
        train_model(train,train)
        #with open(FT_data[args.task] + "test_1M.pkl", 'rb') as f:
            # format of train, test and dev
            # dict <class 'dict'>:{
            # 'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
            # 'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
        print("loading test data...")
        test = np.load(FT_data[args.task] + 'test_1M_optimized.npy', allow_pickle=True)
            #test = pickle.load(f, encoding='ISO-8859-1')

        eval_model(test)
    else:
        with open(FT_data[args.task] + "test_1M.pkl", 'rb') as f:
            # format of train, test and dev
            # dict <class 'dict'>:{
            # 'y' {list} <class 'list' (int)>   (for train 1000000, test and dev 5000000)  #label
            # 'cr' {list} <class 'list' (int)>             #consequtive context and response with differnet size
            print("loading test data...")
            #test = pickle.load(f, encoding='ISO-8859-1')
        eval_model(train)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")

'''
