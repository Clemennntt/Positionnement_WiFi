import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#### Préparation des données ####

# Charge dataset 
dataset = pd.read_csv('fichier.csv')

# Visualisation (comme tu as fait avec les couleurs ???)
##A RAJOUTER##

# Split the data
X = dataset[['position_x', 'position_y', 'adresse_mac']]
y = dataset['DB_mean']

# Sélection des colonnes à normaliser et normalisation
columns_to_normalize = ['position_x', 'position_y']
scaler = StandardScaler()
X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])

# Split en test et train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#### Modèle et prédiction ####

# KNN model & prédiction
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#### Evaluation ####

# Métriques d'éval (cf TP1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Square Error (RMSE):", rmse)
print("R² Score:", r2)


# Affichaged es prédictions
predictions = pd.DataFrame({'Position_x': X_test['position_x'], 'Position_y': X_test['position_y'], 'Adresse_MAC': X_test['adresse_mac'], 'DB_mean_prédit': y_pred})
print(predictions)
