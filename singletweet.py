import os
import sys
from collections import defaultdict
from sklearn import tree
import numpy as np
import gensim
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import json
from joblib import load


def load_single_file():
    CURRENT_PATH = sys.argv[1]
    f = open(CURRENT_PATH, "r")
    data = f.read()
    f.close()
    final = []
    final.append(data)

    return final


def main():
    tweet = load_single_file()
    print(str(tweet))
    model, prediction = svc_classifier(tweet)
    if prediction == 1:
        output = {
            "Status": 200,
            "Prediction": 1,
            "Label": "Offensive"
        }
    else:
        output = {
            "Status": 200,
            "Prediction": 0,
            "Label": "Not Offensive"
        }

    print(json.dumps(output))
    return model


def svc_classifier(tweet):
    model = load("svc_tfidf.joblib")
    prediction = model.predict(tweet)
    print(prediction)
    
    return model, prediction


model = main()
