from transformers import BertTokenizer
import pickle
from tqdm import tqdm
import pickletools
import numpy as np
from input_params import *

def FT_data(file, maxlen, tokenizer=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    with open(file, 'r') as myfile:
        data = [next(myfile) for x in range(maxlen)]#open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data[0:]]  #list of list of string (the same format as in my code)
    y = [int(a[0]) for a in data]  #labels: list of int
    cr = [ [sen for sen in a[1:]] for a in data] #list of list of strings (the last string in each list is response)
    cr_list=[]
    cnt=0
    for s in tqdm(cr):
        s_list=[]
        for sen in s[:-1]:
            if len(sen)==0:
                cnt+=1
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    print(cnt)
    return y, cr_list

def PT_data():
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(data_file_path + 'train.txt', 'r').readlines()[0:1000]
    data = [sent.split('\n')[0].split('\t') for sent in data[0:]]
    y = [int(a[0]) for a in data]
    cr = [[sen for sen in a[1:]] for a in data]
    crnew=[]
    for i,crsingle in enumerate(cr):
        if y[i]==1:
            crnew.append(crsingle)
    crnew=crnew
    pickle.dump(crnew, file=open(data_file_path+"ubuntu_post_train_00M.pkl", 'wb'))


if __name__ == '__main__':
    #Fine_tuning data constuction
    #including tokenization step
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)

    train, test, dev = {}, {}, {}
    train['y'], train['cr'] = FT_data(data_file_path+'train.txt', 1000, tokenizer=bert_tokenizer)
    dev['y'], dev['cr'] = FT_data(data_file_path+'valid.txt', 1000, tokenizer=bert_tokenizer)
    test['y'], test['cr']= FT_data(data_file_path+'test.txt', 1000, tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, dev, test
    pickle.dump(dataset, open(data_file_path+'dataset_00.pkl', 'wb'))

    #posttraining data construction
    #does not include tokenization step
    PT_data()


# if __name__ == '__main__':
#     #Fine_tuning data constuction
#     #including tokenization step
#     bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
#     special_tokens_dict = {'eos_token': '[eos]'}
#     num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
#
#     train, test, dev = {}, {}, {}
#     train['y'], train['cr'] = FT_data('ubuntu_data/train.txt', tokenizer=bert_tokenizer)
#     #dev['y'], dev['cr'] = FT_data('ubuntu_data/valid.txt', tokenizer=bert_tokenizer)
#     #test['y'], test['cr']= FT_data('ubuntu_data/test.txt',tokenizer=bert_tokenizer)
#     #char_vocab = defaultdict(float)
#     #dataset = train, dev, test
#     # pickled_train = pickle.dumps(train)
#     # optimized_train = pickletools.optimize(pickled_train)
#     # pickle.dump(optimized_train, open('ubuntu_data/train_1M_optimized.pkl', 'wb'))
#     # pickled_dev = pickle.dumps(dev)
#     # optimized_dev = pickletools.optimize(pickled_dev)
#     # pickle.dump(optimized_dev, open('ubuntu_data/valid_1M_optimized.pkl', 'wb'))
#     # pickled_test = pickle.dumps(test)
#     # optimized_test = pickletools.optimize(pickled_test)
#     # pickle.dump(optimized_test, open('ubuntu_data/test_1M_optimized.pkl', 'wb'))
#     np.save('ubuntu_data/train_1M_optimized.npy', train)
#     #posttraining data construction
#     #does not include tokenization step
#     # modify the default parameters of np.load
#     #np.load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
#     #a = np.load('ubuntu_data/test_1M_optimized.npy',allow_pickle=True)
# #    PT_data()
# print("end")