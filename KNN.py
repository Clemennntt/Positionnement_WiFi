import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from traitement_data import prepare_data, pivot_dataset, convert_same_SA, filtrage_colonne_for_ML, correlation_Pearson, correlation_Spearman, khi2, codage_one_hot_for_ML
from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import pickle


import warnings
# Ignorer les avertissements
warnings.filterwarnings("ignore")


def test_correlation(data) :
    """ 
    On fait les différents tests
    """
    correlation_Pearson(data)
    correlation_Spearman(data)
    khi2(data)

def __prepare_data(data_train, data_test):
    #### Prepare the data ####
    df_train_pivoted = pivot_dataset(data_train)
    df_test_pivoted = pivot_dataset(data_test)

    # Split the data
    X_train = df_train_pivoted.drop(['x', 'y'], axis=1).to_numpy()
    y_train = df_train_pivoted[['x', 'y']].to_numpy()

    X_test = df_test_pivoted.drop(['x', 'y'], axis=1).to_numpy()
    y_test = df_test_pivoted[['x', 'y']].to_numpy()


    # Codage one-hot des variables SA
    #X_train = codage_one_hot_for_ML(X_train)
    #X_test = codage_one_hot_for_ML(X_test)
    return X_train, y_train, X_test, y_test

def __create_and_train_model(X_train, y_train, X_test, y_test):
    #### Modèle et prédiction ####
    from sklearn.model_selection import KFold

    knn = KNeighborsRegressor(n_neighbors=3)
    kfold = KFold(n_splits=5, shuffle=True)
    best_model = None
    best_metric = float('inf')
    for train_index, val_index in kfold.split(X_train):
        # KNN model & prédiction
        X_train_fold = X_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        y_val_fold = y_train.iloc[val_index]

        knn.fit(X_train, y_train)
        y_pred = estimate_position(knn, X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_metric:
            best_metric = rmse
            best_model = knn

    # # KNN model & prédiction
    # knn = KNeighborsRegressor(n_neighbors=3)
    # knn.fit(X_train, y_train)
    return knn

def estimate_position(knn, X_test):
    y_pred = knn.predict(X_test)
    return y_pred

def __evaluate_model(knn, X_test, y_test):
    #### Evaluation ####
    y_pred = estimate_position(knn, X_test)

    # Métriques d'éval
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error (RMSE):", rmse)
    print("R² Score:", r2)

    return
    # après ça bug
    #### Affichage des prédictions ####

    predictions = pd.DataFrame({'DB_mean_recu': X_test['RSSI'].values,
                                'x_predi': y_pred[:, 0],
                                'y_predi': y_pred[:, 1],
                                'z_predi': y_pred[:, 2],
                                'x_vrai': y_test['x'].values,
                                'y_vrai': y_test['y'].values,
                                'z_vrai': y_test['z'].values})

    print('Voici le tableau avec les prédictions :')                            
    print(predictions.head(10))

    #### Graphes ####

    residus = y_test - y_pred

    #---- 1.Diagramme de dispersion des résidus ----#

    # Conversion de y_pred en DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=['x', 'y', 'z'])
    residus_x = residus['x']
    residus_y = residus['y']
    residus_z = residus['z']
    plt.scatter(y_pred_df['x'], residus_x, color='red', label='x')
    plt.scatter(y_pred_df['y'], residus_y, color='green', label='y')
    plt.scatter(y_pred_df['z'], residus_z, color='blue', label='z')

    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.title('Diagramme de dispersion des résidus par dimension')
    plt.legend()
    plt.show()

    #---- 2.Distribution des résidus ----#
    
    sns.histplot(residus, kde=True)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus')
    plt.show()

    #---- 3.Conversion des résidus en une seule dimension ----#

    residus_sans_z = residus.drop('z', axis = 1)
    residus_sans_z_flatten = residus_sans_z.values.flatten()
    # Distribution des résidus sans la variable z
    sns.histplot(residus_sans_z_flatten, kde=True)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus (sans la variable z)')
    plt.show()

def __save_model(knn, path):
    """
    Private methode 
    Save the trained model to file
    """
    with open(path, 'xb') as f:
        pickle.dump(knn, f)

def load_model(path):
    """
    Load the tree model from source
    """
    with open(path, 'rb') as f:
        knn = pickle.load(f)
    return knn



#========================================================================================================================================#
#================================================================= Main =================================================================#
#========================================================================================================================================#


def main(argc, argv):
    """
    Fonction si le fichier et run directement. Si aucun params ne sont passés, des fichiers par défaut sont chargés
    """
    train_set_path = 'salle_P0_16_06_B2_filtered.csv'
    test_set_path = 'salle_P0_16_06_A2_filtered.csv'
    if argc >= 3:
        train_set_path = argv[1]
        test_set_path = argv[2]

    
    #### Préparation des données ####
    data_train, data_test = prepare_data(train_set_path, test_set_path)

    X_train, y_train, X_test, y_test = __prepare_data(data_train, data_test)


    #### Evaluation ? ####
    """# On garde les colonnes essentiels : SA, COUNT, x, y, z
    data_train_ML = filtrage_colonne_for_ML(data_train)
    data_test_ML = filtrage_colonne_for_ML(data_test)

    # Tests de corrélation
    test_correlation(data_train_ML)"""


    #### KNN training #### 
    """knn = __create_and_train_model(X_train, y_train)
    __evaluate_model(knn, X_test, y_test)
    __save_model(knn, 'model/KNN_model.pkl')"""


    #### example ####
    knn = load_model('model/KNN_model.pkl')

    Fingerprint_to_test0 = X_test[12]
    Fingerprint_to_test1 = X_test[13]
    Fingerprint_to_test2 = X_test[14]
    
    print(Fingerprint_to_test0)

    pred = estimate_position(knn, [Fingerprint_to_test0])
    print(pred)

    preds = estimate_position(knn, [Fingerprint_to_test0, Fingerprint_to_test1, Fingerprint_to_test2])
    print(preds)


if __name__ == "__main__":
    argc = len(sys.argv)
    main(argc, sys.argv)