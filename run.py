import os
from collections import defaultdict
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


def main():
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
    for label in data:
        print('For label ', label, ' we have ', len(data[label]), ' documents')

    # temp = load_files(data[one_file])
    # print(temp)

    label2id = {'train_set.tsv': 0}
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
    # temp1 = X.toarray()

    # categories = ['OFF', 'NOT']
    # temp2 = categories.toarray()
    # print(temp1.shape)
    # print(temp2.shape)
    # docs_train, docs_test, y_train, y_test = train_test_split(
    #     temp1, temp2, test_size=0.5)

    # clf = Pipeline([
    #     ('vec', vectorizer),
    #     ('clf', tree.DecisionTreeClassifier()),
    # ])
    # Fit a decision tree (classifier based on a set of if-else questions to eventually make an informed decision)
    clf = tree.DecisionTreeClassifier()
    # The fit method takes two equally long arrays, one with data points (X), and one with labels (all_labels).
    # By convention you will see the labels referred to as 'y', and data as 'X'.
    clf = clf.fit(X, all_labels)

    # Remember the type of data we are dealing with
    # for l in data:
    #     for tweet in data[l]:
    #         print(l, '->', tweet.strip())

    y_predicted = clf.predict(X)
    print(y_predicted)


main()
