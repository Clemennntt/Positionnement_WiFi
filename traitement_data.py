#! /usr/bin/python3
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
import sys

def _mac_to_int(mac_address):
    """
    Permet de coder les SA (string) en valeur numérique (int) en conservant toute l'information 
    """
    hex_parts = mac_address.split(':')
    hex_string = ''.join(hex_parts)
    return int(hex_string, 16)

def codage_one_hot_for_ML(data) :
    """
    Permet de coder les SA en one hot vu que c'est pas des valeurs numérique
    """
    encoder = OneHotEncoder()

    # Calcul du nbr de SA diff
    nombre_sa_differentes = data['SA'].nunique()
    print('--------------------')
    print("Nombre de SA différentes dans le dataset :", nombre_sa_differentes)

    # Ajout des données en one hot avec concatenate
    encoded_SA = encoder.fit_transform(data[['SA']]).toarray()
    data = pd.concat([data, pd.DataFrame(encoded_SA)], axis=1)
    
    # Affichage du new dataset
    print(data.head(3))

    return data

def _extract_clean_dataset(filename):
    """
    Charge un fichier et le nettoie. (sauf les non utf8 genre les emoji ça faut le faire à la main avec libre calc psk jsp pk ça marche pô)
    """
    df = pd.read_csv(filename)
    df.columns = df.columns.map(lambda x: x.strip())
    
    return df

def _dataset_groupby_saxyz(df):
    """
    Réduit la quantitée de données et fait le tri des colonnes à garder
    """
    df2 = df.groupby(['SA', 'x', 'y', 'z']).aggregate({
        'RSSI': 'mean',
        'COUNT': 'sum',
        'SSID': 'unique', 
        'BSSID': 'unique', 
        'DA': 'unique', 
        'antenna_index': 'unique', 
        'channel': 'unique', 
        'TSFT': 'unique',
        'flags': 'unique', 
        'data_rate': 'unique', 
        'rx_flag': 'unique', 
        'timestamp': 'median',
        'mcs': 'unique',
        'ampdu_status': 'unique',
    }).reset_index()

    return df2

def pivot_dataset(df):
    """
    Fait pivoter la table pour avoir xy en première colonne puis chaque SA en colonne (et ajoute -99 pour les vals manquantes)
    """
    copy = df.copy()
    copy['SA'] = copy['SA'].apply(_mac_to_int)

    pivoted = pd.pivot_table(copy, values='RSSI', index=['x', 'y'], columns=['SA'], fill_value=-99).reset_index()

    return pivoted

def prepare_data(train_set_path, test_set_path):
    """
    Réalisé toute les opération de netoyage et de tri. Les datasets peuvent alors être utilisés
    """
    # load datasets and clean data if needed (A manual deletion of non UTF-8 char should be done manually since pandas doesn't seems to find them)
    df_train = _extract_clean_dataset(train_set_path)
    df_test = _extract_clean_dataset(test_set_path)

    df_train_cleaned, df_test_cleaned = convert_same_SA(df_train, df_test)

    # perform data reduction
    df_train_converted = _dataset_groupby_saxyz(df_train_cleaned)
    df_test_converted = _dataset_groupby_saxyz(df_test_cleaned)

    return df_train_converted, df_test_converted

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
    data1_final = data1[data1['SA'].isin(inter_SA)]
    data2_final = data2[data2['SA'].isin(inter_SA)]

    return data1_final, data2_final

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





def main(argc, argv):
    """
    Fonction si le fichier et run directement. Si aucun params ne sont passés, des fichiers par défaut sont chargés
    """
    train_set_path = 'salle_P0_16_06_A3_ready_to_use.csv'
    test_set_path = 'salle_P0_16_06_B3_ready_to_use.csv'
    if argc >= 3:
        train_set_path = argv[1]
        test_set_path = argv[2]

    # perform data reduction
    df_train_converted, df_test_converted = prepare_data(train_set_path, test_set_path)

    #df_train_converted.to_csv(path_or_buf="train_set.csv",index=False)
    #df_test_converted.to_csv(path_or_buf="test_set.csv",index=False)

    df_train_pivoted = pivot_dataset(df_train_converted)
    df_test_pivoted = pivot_dataset(df_test_converted)

    df_train_pivoted.to_csv(path_or_buf="train_set_pivoted.csv",index=False)
    df_test_pivoted.to_csv(path_or_buf="test_set_pivoted.csv",index=False)

if __name__ == "__main__":
    """
    Pas toucher =3
    """
    argc = len(sys.argv)
    main(argc, sys.argv)