import os
from collections import defaultdict


def main():
    # Edit this to change which model to use
    CURRENT_PATH = "train_set"
    # read each file, add that file to a list of files belonging to the same class
    data_path = CURRENT_PATH
    # get all files contained in the directory
    list_of_files = os.listdir(data_path)
    list_of_files.pop(0)
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

    label2id = {'dev_set.tsv': 2, 'train_set.tsv': 1, 'trial_set.tsv': 0}
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

    print(len(all_documents), type(all_documents))

    # Here we start using scikit-learn!
    # the CountVectorizer can be used to transform each document into a 'bag-of-words' representation
    # https://en.wikipedia.org/wiki/Bag-of-words_model
    # Each document is then represented as *presence* or *absence* of the words in our 'bag'
    from sklearn.feature_extraction.text import CountVectorizer
    # We don't need to use a dictionary for counting word frequency and selecting the important ones.
    # sklearn has this (and more advanced) built-in functions!
    vectorizer = CountVectorizer(max_features=50, stop_words='english')
    X = vectorizer.fit_transform(all_documents)
    print('These are our "features":', ', '.join(vectorizer.get_feature_names()))
    print('A value of zero means that word is not in the document, one if yes')
    print('Each value in a document array corresponds by position with the above features')
    print(X.toarray())

    # Fit a decision tree (classifier based on a set of if-else questions to eventually make an informed decision)
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    # The fit method takes two equally long arrays, one with data points (X), and one with labels (all_labels).
    # By convention you will see the labels referred to as 'y', and data as 'X'.
    clf = clf.fit(X, all_labels)

    # Remember the type of data we are dealing with
    for l in data:
        for tweet in data[l]:
            print(l, '->', tweet.strip())


main()
