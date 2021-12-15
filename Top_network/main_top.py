import os
import torch
import time
import argparse
import numpy as np
import random
import csv
from config_params import *
from input_params import *
import preprocess_data, train, dual_encoder, evaluate
import mini_net_dual_encoder
import mini_net_smn

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

################################
args, _ = define_args().parse_known_args()
print('HELLO!!   ', device)

train_clusters = preprocess_data.read_clusters(args, "train")
valid_clusters = preprocess_data.read_clusters(args, "valid")
test_clusters = preprocess_data.read_clusters(args, "test")

vocabs = []
models = []
for cluster in range(num_clusters):
    vocab_path = os.path.join(args.dataPath, "mini_networks/vocab_") + str(cluster) + '.txt'
    voc = preprocess_data.load_vocab(vocab_path)
    vocabs.append(voc)
    if args.modelName == 'SMN':
        models.append(mini_net_smn.SMN(
            vocab=voc,
            input_size=embed_dim,  # embedding dim
            hidden_size=hidden_size,  # rnn dim
            vocab_size=len(voc),  # vocab size
            bidirectional=False,  # really should change!
            rnn_type='gru',
            num_layers=1,
            dropout=dropout_rate,
            emb_dir=args.embDir
        ))
    elif args.modelName == 'Dual_GRU':
        models.append(mini_net_dual_encoder.Encoder(
            vocab=voc,
            input_size=embed_dim,  # embedding dim
            hidden_size=hidden_size,  # rnn dim
            vocab_size=len(voc),  # vocab size
            bidirectional=False,  # really should change!
            rnn_type='gru',
            num_layers=1,
            dropout=dropout_rate,
            emb_dir=args.embDir
        ))
    save_mini_model_path = os.path.join(args.dataPath, "mini_networks/saved_model_") + str(cluster) + '.pth'
    # Load the best saved model.
    models[cluster].load_state_dict(torch.load(save_mini_model_path, map_location=torch.device('cpu')))
    models[cluster].to(device)
    models[cluster].eval()

##############################  load data and vocab #################################
print('Loading dataset ...')
train_rows, valid_rows, test_rows, vocab = preprocess_data.load_Data(args)
# print('building dataLoaders ...')
# train_data_loader, valid_data_loader, test_data_loader = preprocess_data_smn.get_data_loaders(train_rows, valid_rows, test_rows, args, device)
vocab_size = len(vocab)

############################ define model ############################################
print(f'model name::  {args.modelName}')
if args.modelName == 'SMN':
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
if args.modelName == 'SMN_PAPER':
    model = smn_paper.SMN(
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
if args.modelName == 'SMN_TEST':
    model = smn_smn.SMN(
        voc_len=vocab_size,  # vocab size
    )

elif args.modelName == 'Dual_GRU':
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
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
loss_fn = torch.nn.BCELoss()  #binary cross entropy loss
# pos_weight = torch.FloatTensor([10 - 1]).to(device)
# loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

loss_fn.to(device)


 #####################################################################################
 ###*****************************  run model **************************************###
 #####################################################################################
best_val_metric = None
losses = [1]

################################## train and validation ##############################
for epoch in range(num_epochs):

    ## shuffle train data for each epoch
    #random.shuffle(train_rows)

    epoch_start_time = time.time()
    model, train_losses = train.train(model, loss_fn, optimizer, train_rows, batch_size, epoch, num_epochs, vocab, device, args, vocabs, models, train_clusters)

    with torch.no_grad():
        val_R1, val_R2, val_R5, acc = evaluate.evaluate(model, valid_rows, evaluate_batch_size, epoch, num_epochs, vocab, device, args, vocabs, models, valid_clusters)

    print('-' * 80)
    description = (
        'Valid: [end of epoch{:3d}|time: {:5.2f}s| R1:{:.3f}| R2:{:.3f}| R5:{:.3f}| Acc:{:.3f}'.format(
            epoch + 1,
            time.time() - epoch_start_time,
            val_R1,
            val_R2,
            val_R5,
            acc
        ))
    print(description)
    print('-' * 80)

    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_metric or val_R1 > best_val_metric:
        torch.save(model.state_dict(), os.path.join(args.dataPath,"saved_model.pth"))
        best_val_metric = val_R1

################################## test #################################################
# Load the best saved model.
model.load_state_dict(torch.load(os.path.join(args.dataPath,"saved_model.pth")))
model.to(device)

# Run on test data.
test_R1, test_R2, test_R5, test_acc = evaluate.evaluate(model, test_rows, evaluate_batch_size, 0, num_epochs, vocab, device, args, vocabs, models, test_clusters)

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
