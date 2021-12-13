import os
import torch
import time
import argparse
import numpy as np
import random

from config_params_mini import *
from input_params import *
import preprocess_data, train, evaluate, dual_encoder, smn

#########################  Device configuration ###################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seeds(0)

#########################  set params ###################################
def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataPath', default=data_file_path)
    parser.add_argument('-trim', default=min_freq)
    parser.add_argument('-maxUttNum', default=max_conv_utt_num)
    parser.add_argument('-maxUttLen', default=max_utterance_length)
    parser.add_argument('-dataset', default=dataset)
    parser.add_argument('-doClean', default=DO_ClEAN)
    parser.add_argument('-isHier', default=IS_HIERARCHICAL)
    parser.add_argument('-modelName', default=model_name)
    parser.add_argument('-batchSize', default=batch_size)
    parser.add_argument('-embDir', default=embedding_file_path)

    return parser

args, _ = define_args().parse_known_args()

print('HELLO!!')


def run_mini_network(mini_net_params):
    ##############################  load data and vocab #################################
    print('Loading dataset ...')
    train_rows, valid_rows, test_rows, vocab = preprocess_data.load_Data(args, mini_net_params)
    vocab_size = len(vocab)

    ############################ define model ############################################
    print(f'model name::  {args.modelName}')

    if args.modelName == 'Dual_GRU':
        model = dual_encoder.Encoder(
            vocab=vocab,
            input_size=embed_dim,  # embedding dim
            hidden_size=hidden_size,  # rnn dim
            vocab_size=vocab_size,  # vocab size
            bidirectional=False,  # really should change!
            rnn_type='gru',
            num_layers=1,
            dropout=dropout_rate,
            emb_dir=args.embDir
        )
    elif args.modelName == 'SMN':
        model = smn.SMN(
            vocab=vocab,
            input_size=embed_dim,  # embedding dim
            hidden_size=hidden_size,  # rnn dim
            vocab_size=vocab_size,  # vocab size
            bidirectional=False,  # really should change!
            rnn_type='gru',
            num_layers=1,
            dropout=dropout_rate,
            emb_dir=args.embDir
        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
    loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
    # pos_weight = torch.FloatTensor([10 - 1]).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    loss_fn.to(device)

    #####################################################################################
    ###*****************************  run model **************************************###
    #####################################################################################
    best_val_metric = None
    losses = [1]

    ################################## train and validation ##############################
    for epoch in range(1):#num_epochs):

        ## shuffle train data for each epoch
        # random.shuffle(train_rows)

        epoch_start_time = time.time()
        model, train_losses = train.train(model, loss_fn, optimizer, train_rows, batch_size, epoch, num_epochs, vocab,
                                          device, '', args)

        with torch.no_grad():
            valid_loss = evaluate.evaluate(model, loss_fn, valid_rows, evaluate_batch_size, epoch, num_epochs,
                                           vocab, max_conv_utt_num, max_utterance_length, device,
                                                            '', args, False)

        print("################ validation loss:: " + str(valid_loss) + " ####################")
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_metric or valid_loss < best_val_metric:
            torch.save(model.state_dict(), mini_net_params['save_model_path'])
            best_val_metric = valid_loss

    ################################## test #################################################
    # Load the best saved model.
    model.load_state_dict(torch.load(mini_net_params['save_model_path']))
    model.to(device)

    # Run on test data.
    test_R1, test_R2, test_R5, test_acc = evaluate.evaluate(model, loss_fn, test_rows, evaluate_batch_size, 0, num_epochs, vocab,
                                                            max_conv_utt_num, max_utterance_length, device,
                                                            '', args, True)

    print('=' * 89)
    description = (
        'Test: [R1: {:.3f} | R2: {:.3f} | R5: {:.3f} | Acc: {:.3f}'.format(
            test_R1,
            test_R2,
            test_R5,
            test_acc
        ))
    print(description)
    print('=' * 89)

################ for hierarchical network ##################################
for cluster in range(num_clusters):
    print("Learning network::: [ " + str(cluster) + " ] started ...")
    out_mini_path = os.path.join(args.dataPath, "mini_networks_20/")
    in_mini_path = os.path.join(args.dataPath, "aug_data_20/")
    mini_net_params = {
    "train_path" : in_mini_path + "train_" + str(cluster) + '.csv',
    "valid_path" : in_mini_path + "valid_" + str(cluster) + '.csv',
    "test_path" : in_mini_path + "test_" + str(cluster) + '.csv',
    "save_model_path" : out_mini_path + "saved_model_" + str(cluster) + '.pth',
    "vocab_path" : out_mini_path + "vocab_" + str(cluster) + '.txt'
    #"valid_results_path" : out_mini_path + "valid_scores_" + str(cluster) + '.txt',
    #"test_results_path" : out_mini_path + "test_scores_" + str(cluster) + '.txt'
    }
    run_mini_network(mini_net_params)
#################################################################################################
#################################################################################################
