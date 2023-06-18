import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from traitement_data import convert_same_SA, filtrage_colonne_for_ML, correlation_Pearson, correlation_Spearman, khi2
from scipy.stats import chi2_contingency

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

    # # Split the data
    # X = data_train[['position_x', 'position_y', 'adresse_mac']]
    # y = data_train['DB_mean']

    # # Sélection des colonnes à normaliser et normalisation
    # columns_to_normalize = ['position_x', 'position_y']
    # scaler = StandardScaler()
    # X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])

    # # Split en test et train
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # #### Modèle et prédiction ####

    # # KNN model & prédiction
    # knn = KNeighborsRegressor(n_neighbors=3)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)

    # #### Evaluation ####

    # # Métriques d'éval (cf TP1)
    # mae = mean_absolute_error(y_test, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)

    # print("Mean Absolute Error (MAE):", mae)
    # print("Root Mean Square Error (RMSE):", rmse)
    # print("R² Score:", r2)


    # # Affichaged es prédictions
    # predictions = pd.DataFrame({'Position_x': X_test['position_x'], 'Position_y': X_test['position_y'], 'Adresse_MAC': X_test['adresse_mac'], 'DB_mean_prédit': y_pred})
    # print(predictions)


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