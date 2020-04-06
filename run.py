import os
from collections import defaultdict
from sklearn import tree
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import gensim
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def split_off_not():
    # Edit this to change which model to use
    CURRENT_PATH = "train_set"
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

    for l in data:
        for tweet in data[l]:
            if tweet[-6] == "1":
                f = open("off.txt", "a")
                f.write(tweet)
                f.close()
                print(tweet[-6] + " OFF -> " + tweet)
            else:
                f = open("not.txt", "a")
                f.write(tweet)
                f.close()
                print(tweet[-6] + " NOT -> " + tweet)


def load_file(folder):
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
    train_docs, train_labels = load_file('train_off_not')
    dev_docs, dev_labels = load_file('dev_off_not')
    # vectorizer = CountVectorizer(max_features=50, stop_words='english')
    # X = vectorizer.fit_transform(model)
    # print('These are our "features":', ', '.join(vectorizer.get_feature_names()))
    # create_word2vec(all_documents)
    # X = load_word2vec()
    # decision_tree(train_docs, train_labels, dev_docs, dev_labels)
    # random_forest(train_docs, train_labels, dev_docs, dev_labels)
    # svc_classifier(train_docs, train_labels, dev_docs, dev_labels)
    # logistic_regression(train_docs, train_labels, dev_docs, dev_labels)
    svc_tfidf(train_docs, train_labels, dev_docs, dev_labels)


def svc_tfidf(train_docs, train_labels, dev_docs, dev_labels):
    model = Pipeline([
        ("tf idf vectorizer", TfidfVectorizer(ngram_range=(1, 15), analyzer='char')),
        ("svc", SVC())])
    model.fit(train_docs, train_labels)
    print(model["tf idf vectorizer"])
    # print(model["tf idf vectorizer"].get_feature_names())
    print(classification_report(model.predict(dev_docs), dev_labels))
    report = classification_report(model.predict(dev_docs), dev_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    write_to_csv(str(model["svc"]), df)


def decision_tree(train_docs, train_labels, dev_docs, dev_labels):
    w2v = load_word2vec()
    model = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("decision tree", tree.DecisionTreeClassifier())])
    model.fit(train_docs, train_labels)
    print(classification_report(model.predict(dev_docs), dev_labels))
    report = classification_report(model.predict(dev_docs), dev_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    write_to_csv(str(model["decision tree"]), df)


def random_forest(train_docs, train_labels, dev_docs, dev_labels):
    w2v = load_word2vec()
    model = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("random forest", RandomForestClassifier())])
    model.fit(train_docs, train_labels)
    print(classification_report(model.predict(dev_docs), dev_labels))
    report = classification_report(model.predict(dev_docs), dev_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    write_to_csv(str(model["random forest"]), df)


def svc_classifier(train_docs, train_labels, dev_docs, dev_labels):
    w2v = load_word2vec()
    model = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("svc", SVC())])
    model.fit(train_docs, train_labels)
    print(classification_report(model.predict(dev_docs), dev_labels))
    report = classification_report(model.predict(dev_docs), dev_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    write_to_csv(str(model["svc"]), df)


def logistic_regression(train_docs, train_labels, dev_docs, dev_labels):
    w2v = load_word2vec()
    model = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("logistic regression", LogisticRegression())])
    model.fit(train_docs, train_labels)
    print(classification_report(model.predict(dev_docs), dev_labels))
    report = classification_report(model.predict(dev_docs), dev_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    write_to_csv(str(model["logistic regression"]), df)


def decision_tree_old(data, labels):
    y = np.array(labels)
    kf = StratifiedKFold(n_splits=3)
    score_array = []

    for train_index, test_index in kf.split(data, y):
        X_train, X_test, y_train, y_test = data[train_index], data[test_index], y[train_index], y[test_index]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # report = classification_report(clf.predict(X_test), y_test, zero_division=0, output_dict=True)
        # print(report['0']['f1-score'])
        # results.append(report)
        # divider = '-----------------'
        # df = pd.DataFrame(report).transpose()
        # print(df)
        # print(divider)
        score_array.append(precision_recall_fscore_support(y_test, y_pred))

    avg_score = np.mean(score_array, axis=0)
    print(avg_score)
    df = pd.DataFrame(avg_score).transpose()
    write_to_csv(str(clf), df)


def random_forest_old(data, labels):
    y = np.array(labels)
    kf = StratifiedKFold(n_splits=3)
    score_array = []

    for train_index, test_index in kf.split(data, y):
        X_train, X_test, y_train, y_test = data[train_index], data[test_index], y[train_index], y[test_index]
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score_array.append(precision_recall_fscore_support(y_test, y_pred))

    avg_score = np.mean(score_array, axis=0)
    print(avg_score)
    df = pd.DataFrame(avg_score).transpose()
    write_to_csv(str(clf), df)


def create_word2vec(data):
    documents = []
    for i, line in enumerate(data):
        documents.append(gensim.utils.simple_preprocess(line))

    model = gensim.models.Word2Vec(size=20, min_count=2, iter=10)
    model.build_vocab(documents)
    w1 = ["bitch"]
    print(model.wv.most_similar(positive=w1))


def load_word2vec():
    model = gensim.models.KeyedVectors.load_word2vec_format('crosslingual_EN-ES_english_twitter_100d_weighted.txt.w2v')
    w1 = ["bitch"]
    print(model.wv.most_similar(positive=w1))
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
    return w2v


def write_to_csv(clf, df):
    f = open('results.csv', 'a')
    f.write(clf + '\n')
    f.close()

    df.to_csv('results.csv', mode='a')


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


main()
# split_off_not()
