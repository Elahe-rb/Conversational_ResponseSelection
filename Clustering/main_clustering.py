import csv
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from input_params import *
from config_params import *
import preprocess_data

#########################  set params ###################################
def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataPath', default=data_file_path)
    parser.add_argument('-dataset', default=dataset)
    parser.add_argument('-doClean', default=DO_ClEAN)
    parser.add_argument('-trim', default=min_freq)
    return parser
########

def building_local_datasets(data_rows, cluster_rows, args, out_file):

    out_files = {}
    for num in range(num_clusters):
        out_files[num] = open(os.path.join(args.dataPath+"/aug_data_20", (out_file + "_" + str(num) + ".csv")), "w")

    counter = 0
    utt_counter = 0
    for row, c_row in zip(data_rows, cluster_rows):
        label = row[0]
        context = row[1:-1]  # -1 is for ignoring the last one which is empty #change to eot for run in cuda and uncomment clean_data
        response = row[-1]  # add response utt to the list
        context_clusters = c_row[1:-1]  # do not consider response id into this list
        response_cluster = c_row[-1]

        if not(len(context) == len(context_clusters) or label == c_row[0]):
            print("error!!")
        new_row = label + ","
        doWrite = False
        for utt, c_utt in zip(context, context_clusters):
            if c_utt == response_cluster:
                utt_counter += 1
                new_row += utt + ","
                doWrite = True
        new_row += response
        if doWrite:
            counter+=1
            out_files.get(int(response_cluster)).write(new_row+"\n")
    print("num_augmented_"+out_file+"_data::"+str(counter))
    print("num_selected_" + out_file + "_utt::" + str(utt_counter+1))   #+1 is for response utt

def read_cluster_file(args,filename):
    reader = csv.reader(open(os.path.join(args.dataPath, filename + "_clusters.csv")))
    cluster_rows = list(reader)[:]
    return cluster_rows

def data_augmentation(args):
    print('Loading dataset ...')
    train_rows, valid_rows, test_rows = preprocess_data.load_Data(args)
    building_local_datasets(train_rows, read_cluster_file(args,"train"), args, "train")
    building_local_datasets(valid_rows, read_cluster_file(args,"valid"), args, "valid")
    building_local_datasets(test_rows, read_cluster_file(args,"test"), args, "test")

def classify_docs(batched_docs, model_classifier, feature_vectoreizer):

    feature_vecs = feature_vectoreizer.transform(batched_docs)
    batched_utt_clusters = model_classifier.predict(feature_vecs)
    return batched_utt_clusters

def save_utts_cluster(rows, clusters, args, filename):
    out_file = open(os.path.join(args.dataPath , (filename + "_" + "clusters_20.csv")), "w")
    i = 0
    for row in rows:
        line = row[0] + ","
        line += ",".join([str(clusters[j]) for j in range(i,(i+len(row)-1))])
        i += len(row)-1
        out_file.write(line+"\n")

def clustering(docs):
    vectorizer = TfidfVectorizer()
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    vector_utts = vectorizer.fit(docs)
    feature_utts = vector_utts.transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    #print(feature_names)
    print(feature_utts.shape) #(4954151, 432571)
    model.fit(feature_utts)
    utts_cluster = model.labels_    #nd array

    # save cluster model to use it as classifier
    # vector_utt
    return model, vector_utts, utts_cluster

def split_dialogs_into_utts(rows):
    all_utterances = []
    [[all_utterances.append(utt) for utt in dialog[1:]] for dialog in rows]
    print("num_utt::"+str(len(all_utterances)))
    return all_utterances

######################################################################################
args, _ = define_args().parse_known_args()

print('HELLO!!')
print('Loading dataset ...')
train_rows, valid_rows, test_rows = preprocess_data.load_Data(args)

print('start clustering ...')
classifier_model, feature_vectorizer, train_clusters = clustering(split_dialogs_into_utts(train_rows))
save_utts_cluster(train_rows, train_clusters, args, "train")
valid_clusters = classify_docs(split_dialogs_into_utts(valid_rows), classifier_model, feature_vectorizer)
save_utts_cluster(valid_rows, valid_clusters, args, "valid")
test_clusters = classify_docs(split_dialogs_into_utts(test_rows),classifier_model ,feature_vectorizer)
save_utts_cluster(test_rows, test_clusters,args, "test")

data_augmentation(args)

'''
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
corpus_lsi = lsi[corpus_tfidf] #
'''