#! /usr/bin/python3
import numpy as np
import sys
import math
from sklearn import tree
from traitement_data import prepare_data, pivot_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


def __generate_model(X, Y):
    """
    Private methode 
    Model definition
    """
    #clf = tree.DecisionTreeClassifier()
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X, Y)
    return clf

def __save_model(clf, path):
    """
    Private methode 
    Save the trained model to file
    """
    with open(path, 'xb') as f:
        pickle.dump(clf, f)

def __train_model(df_train):
    """
    Private methode 
    Train the model an return it
    """
    df_train_pivoted = pivot_dataset(df_train)

    X = df_train_pivoted.drop(['x', 'y'], axis=1).to_numpy()
    Y = df_train_pivoted[['x', 'y']].to_numpy()

    clf = __generate_model(X, Y)

    return clf


def __test_model(clf, df_test):
    """
    Private methode 
    Test the model on a full dataset. Print results and return predictions.
    """
    df_test_pivoted = pivot_dataset(df_test)
    X = df_test_pivoted.drop(['x', 'y'], axis=1).to_numpy()
    Y = df_test_pivoted[['x', 'y']].to_numpy()

    y = clf.predict(X)
    dist = [] 
    for i in range(0, len(y)):
        dist.append(math.sqrt((y[i][0] - Y[i][0])**2 + (y[i][1] - Y[i][1])**2))
    
    print(np.mean(dist))


    #### Evaluation ####

    # Métriques d'éval
    mae = mean_absolute_error(Y, y)
    rmse = np.sqrt(mean_squared_error(Y, y))
    r2 = r2_score(Y, y)

    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error (RMSE):", rmse)
    print("R² Score:", r2)


    return y

def load_model(path):
    """
    Load the tree model from source
    """
    with open(path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def estimate_position(clf, X):
    """
    Evaluate a position given a fingerprint
    """
    y = clf.predict(X)
    #y = clf.predict_proba(X)
    return y

def main(argc, argv):
    if argc < 2:
        print("Please provide at least one file")
        exit(1)

    ### Préparation
    # pre preapare data for train and test
    df_train_cleaned, df_test_cleaned = prepare_data(argv[1], argv[2])
    
    """ # train and save model
    clf = __train_model(df_train_cleaned)
    __test_model(clf, df_test_cleaned)
    save_model(clf, 'model/decision_tree_model.pkl')"""# à conserver




    ### Exemple d'utilisation
    # load model from file
    clf2 = load_model('model/decision_tree_model.pkl')
    
    # do this to extract only one line (only one fingerprint to evaluate)
    df_test_pivoted = pivot_dataset(df_test_cleaned)
    Fingerprint_to_test0 = df_test_pivoted.drop(['x', 'y'], axis=1).to_numpy()[12]
    Fingerprint_to_test1 = df_test_pivoted.drop(['x', 'y'], axis=1).to_numpy()[13]
    Fingerprint_to_test2 = df_test_pivoted.drop(['x', 'y'], axis=1).to_numpy()[14]
    
    pred = estimate_position(clf2, [Fingerprint_to_test0]) # juste une pred
    print(pred)
    preds = estimate_position(clf2, [Fingerprint_to_test0, Fingerprint_to_test1, Fingerprint_to_test2]) # plusieurs preds
    print(preds)


if __name__ == "__main__":
    argc = len(sys.argv)
    main(argc, sys.argv)