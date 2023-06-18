import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder

def convert_same_SA(data1, data2) :
    """
    Permet de filtrer les 2 datasets en gardant uniquement les mêmes SA
    """
    # Set contenant les valeurs uniques de SA
    data1_SA = set(data1['SA'])
    data2_SA = set(data2['SA'])

    # data1_SA ∩ data2_SA
    inter_SA = data1_SA.intersection(data2_SA)

    # Filtrage des dataset avec les inter_SA
    data1 = data1[data1['SA'].isin(inter_SA)]
    data2 = data2[data2['SA'].isin(inter_SA)]

    # Affichage de la tail (voir si c'est le même et voir la diff de nbr total de SA)
    print('Voici la conversion avec les mm SA :')
    print(data1.tail(5))
    print(data2.tail(5))

    return data1, data2

def filtrage_colonne_for_ML(data) :
    """
    Permet de garder les colonnes utiles pour les programmes ML
    SA, COUNT, x, y, z
    """
    # Selection des colonnes
    colonnes_useful = ['x', 'y', 'z', 'SA', 'RSSI']
    data = data[colonnes_useful]

    # Afficahge de la head voir le tableau
    print('\nVoici le filtrage avec les colonnes utiles pour le ML :')
    print(data.head(5))
    return data

def correlation_Pearson(data) :
    """
    Permet de faire le test de corrélation de Pearson(linéaire) pour voir la prédictibilité
    """
    # Calcule de la correlation et affichage (Pearson uniquement pour les val numériques)
    correlation = data.corr()
    print('\nVoici le test de correlation de Pearson :')
    print(correlation)

def correlation_Spearman(data) :
    """
    Permet de faire le test de corrélation de Spearman (monotone) pour voir la prédictibilité
    """
    # Calcule de la correlation et affichage (Sspearman uniquement pour les val numériques)
    correlation = data.corr(method='spearman')
    print('\nVoici le test de correlation de Spearman:')
    print(correlation)

def khi2(data) :
    """
    Permet de test le khi2 entre SA et chaque autre variable catégorielle
    """
    #Test entre respectivement x, y, z et SA par rapport à RSSI
    for colonne in ['x', 'y', 'z']:
        tableau_contingence = pd.crosstab(data[colonne], [data['SA'], data['RSSI']])
        chi2, p_value, dof, expected = chi2_contingency(tableau_contingence)
        print(f"\nTest khi2 entre {colonne} et SA par rapport à RSSI:")
        print("Tableau de contingence :")
        print(tableau_contingence)
        print("p-value :", p_value)
        print("")

    # Ancien test entre SA et x, y, z, RSSI mais faible corrélation et PAS PERTINENT
    # for colonne in data.columns:
    #     if colonne != 'SA':
    #         tableau_contingence = pd.crosstab(data['SA'], data[colonne])
    #         chi2, p_value, dof, expected = chi2_contingency(tableau_contingence)
    #         print(f"Test khi2 entre SA et '{colonne}':")
    #         print("Tableau de contingence :")
    #         print(tableau_contingence)
    #         print("p-value :", p_value)
    #         print("")

def codage_one_hot_for_ML(data) :
    """
    Permet de coder les SA en one hot vu que c'est pas des valeurs numérique
    """
    encoder = OneHotEncoder()

    # Calcul du nbr de SA diff
    nombre_sa_differentes = data['SA'].nunique()
    print('--------------------')
    print("Nombre de SA différentes dans le dataset :", nombre_sa_differentes)

    # Ajout des données en one hot 
    encoder.fit(data[['SA']])
    encoded_SA = encoder.transform(data[['SA']]).toarray()

    # Réinitialisation des index des DataFrames
    data = data.reset_index(drop=True)
    encoded_SA_df = pd.DataFrame(encoded_SA)

    # Ajout avec concatenate
    data = pd.concat([data, pd.DataFrame(encoded_SA)], axis=1)

    # drop colonne SA + Nan et affichage
    data = data.drop('SA', axis=1)
    data = data.dropna()

    return data