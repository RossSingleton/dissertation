import os
import sys
from collections import defaultdict
from sklearn import tree
import numpy as np
import gensim
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import json


def load_single_file():
    CURRENT_PATH = sys.argv[1]
    f = open(CURRENT_PATH, "r")
    data = f.read()
    f.close()
    final = []
    final.append(data)

    return final


def load_train_files(folder):
    # Edit this to change which model to use
    CURRENT_PATH = folder
    # read each file, add that file to a list of files belonging to the same class
    data_path = CURRENT_PATH
    # get all files contained in the directory
    list_of_files = os.listdir(data_path)
    # data is a dictionary where each key is a string, and each value is a list
    # Convenient because we do not have to check if the key exists or not, it will always work
    data = defaultdict(list)
    for one_file in list_of_files:
        # os.path.join joins a path to a folder and a file into one manageable path
        # in windows, something like os.path.join(c:\Users\luis,'file.txt') >>> 'c:\\Users\\luis\\file.txt'
        with open(os.path.join(data_path, one_file), 'r') as f:
            for line in f:
                # each line in each file contains one single document
                data[one_file].append(line)

    for label in data:
        print('For label ', label, ' we have ', len(data[label]), ' documents')

    # temp = load_files(data[one_file])
    # print(temp)

    label2id = {'off.txt': 1, 'not.txt': 0}
    # print(data['dev_set.tsv'][1])

    # we will also store all documents in a single array to use the following bit of code
    all_documents = []
    # we will do the same for labels (for later)
    all_labels = []
    # for each label in the dictionary (key)
    for label in data:
        # for each document in the list of documents pertaining to that label
        for document in data[label]:
            # add that document to the array with all documents
            all_documents.append(document)
            all_labels.append(label2id[label])

    return all_documents, all_labels


def main():
    train_docs, train_labels = load_train_files('train_off_not')
    tweet = load_single_file()
    model, prediction = svc_classifier(train_docs, train_labels, tweet)
    if prediction == 1:
        output = {
            "Prediction": 1,
            "Label": "Offensive"
        }
    else:
        output = {
            "Prediction": 0,
            "Label": "Not Offensive"
        }

    print(json.dumps(output))
    return model


def svc_classifier(train_docs, train_labels, tweet):
    w2v = load_word2vec()
    model = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("svc", SVC())])
    model.fit(train_docs, train_labels)
    prediction = model.predict(tweet)
    print(prediction)
    
    return model, prediction


def load_word2vec():
    model = gensim.models.KeyedVectors.load_word2vec_format('crosslingual_EN-ES_english_twitter_100d_weighted.txt.w2v')
    w1 = ["bitch"]
    # print(model.wv.most_similar(positive=w1))
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
    return w2v


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # this line is different from python2 version - no more itervalues
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


model = main()
