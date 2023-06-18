import pandas as pd

def convert_same_SA(filename1, filename2) :
    """
    Permet de filtrer les 2 datasets en gardant uniquement les mêmes SA
    """
    # Import les données
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    # Set contenant les valeurs uniques de SA
    data1_SA = set(data1['SA'])
    data2_SA = set(data2['SA'])

    # data1_SA ∩ data2_SA
    inter_SA = data1_SA.intersection(data2_SA)

    # Filtrage des dataset avec les inter_SA
    data1_final = data1[data1['SA'].isin(inter_SA)]
    data2_final = data2[data2['SA'].isin(inter_SA)]

    # Affichage de la tail (voir si c'est le même et voir la diff de nbr total de SA)
    print(data1_final.tail(5))
    print(data2_final.tail(5))

    return data1_final, data2_final


convert_same_SA('salle_P0_16_06_A3_ready_to_use.csv', 'salle_P0_16_06_B3_ready_to_use.csv')