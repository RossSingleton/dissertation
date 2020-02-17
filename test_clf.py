import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


def main():
    data_folder = sys.argv[1]
    dataset = load_files(data_folder)
    print(dataset)
    vectorizer = CountVectorizer(max_features=50, stop_words='english')
    print('These are our "features":', ', '.join(vectorizer.get_feature_names()))
    clf = Pipeline([
        ('vec', vectorizer),
        ('clf', tree.DecisionTreeClassifier()),
    ])

    clf.fit(docs_train, y_test)
    y_predicted = clf.predict(docs_test)
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))


main()
