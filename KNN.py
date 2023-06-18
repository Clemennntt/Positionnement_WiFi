import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from traitement_data import convert_same_SA, filtrage_colonne_for_ML, correlation_Pearson, correlation_Spearman, khi2, codage_one_hot_for_ML
from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
# Ignorer les avertissements
warnings.filterwarnings("ignore")


def test_correlation(data) :
    """ 
    On fait les différents tests
    """
    correlation_Pearson(data_train)
    correlation_Spearman(data_train)
    khi2(data_train)

def KNN(data_train, data_test) :
    """
    Programme KNN
    """
    #### Prepare the data ####

    # Split the data
    X_train = data_train[['SA', 'RSSI']]
    y_train = data_train[['x', 'y', 'z']]
    X_test = data_test[['SA', 'RSSI']]
    y_test = data_test[['x', 'y', 'z']]

    print(X_train.head(3))

    # Codage one-hot des variables SA
    X_train = codage_one_hot_for_ML(X_train)
    X_test = codage_one_hot_for_ML(X_test)

    #### Modèle et prédiction ####

    # KNN model & prédiction
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    #### Evaluation ####

    # Métriques d'éval
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error (RMSE):", rmse)
    print("R² Score:", r2)

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

    




#========================================================================================================================================#
#================================================================= Main =================================================================#
#========================================================================================================================================#

if __name__ == "__main__":
    
    #### Préparation des données ####

    # Import les données
    data_test = pd.read_csv('salle_P0_16_06_A3_ready_to_use.csv')
    data_train = pd.read_csv('salle_P0_16_06_B3_ready_to_use.csv')

    # Filtrage des datas 
    data_train, data_test = convert_same_SA(data_train, data_test)

    ## A PARTIR D'ICI DATA_TRAIN et DATA_TEST SONT LES BASES POUR LES 3 PROGRAMMES (MLx + DL) ##

    #### Programme Machine Learning ####
    #### Filtrage pour ML ####

    # On garde les colonnes essentiels : SA, COUNT, x, y, z
    data_train_ML = filtrage_colonne_for_ML(data_train)
    data_test_ML = filtrage_colonne_for_ML(data_test)

    # Tests de corrélation
    test_correlation(data_train_ML)

    #### Programme 1 : KNN #### 

    KNN(data_train_ML, data_test_ML)