import os
from collections import defaultdict
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np


def split_off_not():
    # Edit this to change which model to use
    CURRENT_PATH = "dev_set"
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


def main():
    # Edit this to change which model to use
    CURRENT_PATH = "off_not"
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

    # Here we start using scikit-learn!
    # the CountVectorizer can be used to transform each document into a 'bag-of-words' representation
    # https://en.wikipedia.org/wiki/Bag-of-words_model
    # Each document is then represented as *presence* or *absence* of the words in our 'bag'
    # We don't need to use a dictionary for counting word frequency and selecting the important ones.
    # sklearn has this (and more advanced) built-in functions!
    vectorizer = CountVectorizer(max_features=50, stop_words='english')
    X = vectorizer.fit_transform(all_documents)
    print('These are our "features":', ', '.join(vectorizer.get_feature_names()))

    # Fit a decision tree (classifier based on a set of if-else questions to eventually make an informed decision)
    # The fit method takes two equally long arrays, one with data points (X), and one with labels (all_labels).
    # By convention you will see the labels referred to as 'y', and data as 'X'.

    y = np.array(all_labels)
    kf = KFold(n_splits=3)

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        print(classification_report(clf.predict(X_test), y_test, zero_division=0))
        print('-----------------')


main()
# split_off_not()
