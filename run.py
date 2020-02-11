import os
from collections import defaultdict


def loadData():
    CURRENT_PATH = "data_set"
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


loadData()
