#! /usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from sklearn import tree
from traitement_data import prepare_data, pivot_dataset, _extract_clean_dataset, convert_same_SA


def train_model(df_train):
    df_train_pivoted = pivot_dataset(df_train)

    X = df_train_pivoted.drop(['x', 'y'], axis=1)
    Y = df_train_pivoted[['x', 'y']].to_numpy()
    clf = generate_model(X, Y)

    return clf

def generate_model(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    return clf

def test_model(clf, df_test):
    df_test_pivoted = pivot_dataset(df_test)
    X = df_test_pivoted.drop(['x', 'y'], axis=1)
    Y = df_test_pivoted[['x', 'y']].to_numpy()

    y = clf.predict(X)
    dist = [] 
    for i in range(0, len(y)):
        dist.append(math.sqrt((y[i][0] - Y[i][0])**2 + (y[i][1] - Y[i][1])**2))
    
    print(np.mean(dist))

    return y

def estimate_position(clf, X):
    y = clf.predict_proba(X)
    return y


def main(argc, argv):
    if argc < 2:
        print("Please provide at least one file")
        exit(1)

    df_train_cleaned, df_test_cleaned = prepare_data(argv[1], argv[2])
    
    clf = train_model(df_train_cleaned)
    
    test_model(clf, df_test_cleaned)

    df_test_pivoted = pivot_dataset(df_test_cleaned)
    #X = df_test_pivoted.drop(['x', 'y'], axis=1)
    #print(X.iloc[0])
    #pred = estimate_position(clf, X.iloc[0])
    #print(pred)


if __name__ == "__main__":
    argc = len(sys.argv)
    main(argc, sys.argv)