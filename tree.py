#! /usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from sklearn import tree

def extract_clean_dataset(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.map(lambda x: x.strip()) #remove spaces, tabs, etc, that could be produced by term
    return df

def generate_model(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    return clf

def test_model(clf, Y):
    return clf.predict(Y)


def main():
    argc = len(sys.argv)
    if argc < 2:
        print("Please provide at least one file")
        exit(1)
    
    df_train = extract_clean_dataset(sys.argv[1])
    X = df_train.drop(['x', 'y'], axis=1)
    Y = df_train[['x', 'y']]
    clf = generate_model(X, Y)

    #print(X[-1:])
    df_test = extract_clean_dataset(sys.argv[2])
    X = df_test.drop(['x', 'y'], axis=1)
    Y = df_test[['x', 'y']]

    x = test_model(clf, X)
    print(x)
    #print(Y)


if __name__ == "__main__":
    main()