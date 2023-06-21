import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from KNN import estimate_position as predict_knn
from traitement_data import pivot_dataset, prepare_data
from tree import estimate_position as predict_tree
from RandomForest import estimate_position as predict_rf
from util import *


class MyApp:
    def __init__(self, master):
        self.master = master
        master.title("Apprentissage automatique")
        
        self.dark = True
        
        self.raw_df = None
        
        self.raw_file = ""
        self.entry_file = ""
        self.dataname = ""
        self.modelline = 0
        
        # Variables pour les entrées de l'utilisateur
        self.raw_file_name = tk.StringVar()
        self.entry_file_name = tk.StringVar()

        # Création des widgets pour l'interface graphique
        self.create_widgets()

    def create_widgets(self):
        """
        Création des widgets pour l'interface graphique
        """
        self.topline()
        self.centerline()
        self.bottomline()
        
    def topline(self):
        """
        Création de la ligne de widget du haut
        """
        # Création du bouton pour changer de mode
        self.mode_button = ctk.CTkSwitch(self.master, text="Mode sombre", command=self.toggle_mode)
        self.mode_button.select()
        self.mode_button.pack(side=ctk.TOP, padx=10, pady=10, anchor='nw')
        
        # Ligne pour la visualisation de données brutes
        raw_data_frame = ctk.CTkFrame(self.master)
        raw_data_frame.pack(side=ctk.TOP, pady=10, padx=20)
        
        raw_data_label = ctk.CTkLabel(raw_data_frame, text="Données brutes:")
        raw_data_label.pack(side=ctk.LEFT, padx=10)
        
        raw_data_entry = ctk.CTkEntry(raw_data_frame, textvariable=self.raw_file_name)
        raw_data_entry.pack(side=ctk.LEFT, padx=10)
        raw_data_entry.configure(state="readonly")
        
        raw_data_search_button = ctk.CTkButton(raw_data_frame, text="Parcourir", command= lambda: self.browse_file("raw"))
        raw_data_search_button.pack(side=ctk.LEFT, padx=10)
        
        self.data_menu = ctk.CTkOptionMenu(raw_data_frame, values=[], command= self.optionmenu_callback)
        self.data_menu.pack(side=ctk.LEFT, padx=10)
        self.data_menu.set("Chargez les données...")
        
        raw_data_visualize_button = ctk.CTkButton(raw_data_frame, text="Visualiser:", command=self.visualize_heat)
        raw_data_visualize_button.pack(side=ctk.LEFT, padx=10)
    
    def centerline(self):
        """
        Création de la ligne de widget du milieu
        """
        # Ligne pour le ML
        ml_frame = ctk.CTkFrame(self.master)
        ml_frame.pack(side=ctk.TOP)
        
        entry_label = ctk.CTkLabel(ml_frame, text="Données d'entrée:")
        entry_label.pack(side=ctk.LEFT, padx=10)
        
        entry_entry = ctk.CTkEntry(ml_frame, textvariable= self.entry_file_name)
        entry_entry.pack(side=ctk.LEFT, padx=10)
        entry_entry.configure(state="readonly")
        
        entry_search_button = ctk.CTkButton(ml_frame, text="Parcourir", command= lambda: self.browse_file("entry"))
        entry_search_button.pack(side=ctk.LEFT, padx=10)
        
        self.model_menu = ctk.CTkOptionMenu(ml_frame, values=[], command= self.modelmenu_callback)
        self.model_menu.pack(side=ctk.LEFT, padx=10)
        self.model_menu.set("Chargez les données...")
        
        select_ml_label = ctk.CTkLabel(ml_frame, text="Prédire avec:")
        select_ml_label.pack(side=ctk.LEFT, padx=10)
        
        mode_ml_frame = ctk.CTkFrame(ml_frame)
        mode_ml_frame.pack(side=ctk.LEFT)
        
        KNN_button = ctk.CTkButton(mode_ml_frame, text="KNN", command= lambda: self.predict("knn"))
        KNN_button.pack(side=ctk.TOP, padx=10)
        
        arbre_button = ctk.CTkButton(mode_ml_frame, text="Arbre de décision", command= lambda: self.predict("tree"))
        arbre_button.pack(side=ctk.TOP, padx=10)
        
        m3_button = ctk.CTkButton(mode_ml_frame, text="M3", command=print(""))
        m3_button.pack(side=ctk.TOP, padx=10)
        
        tous_button = ctk.CTkButton(ml_frame, text="Tous ", command= lambda: self.predict("all"))
        tous_button.pack(side=ctk.LEFT, padx=10)
        
    
    def bottomline(self):
        """
        Création de la ligne de widget du bas
        """
        map_label = ctk.CTkLabel(self.master, text="Carte de la classe:")
        map_label.pack(anchor=ctk.CENTER, pady=10)
        
        on = Image.open("map.png")
        map_image = ctk.CTkImage(dark_image=on, light_image=on, size=(852,402))
        self.map_label = ctk.CTkLabel(self.master, image=map_image, text="")
        self.map_label.pack(anchor=ctk.CENTER)
        

    def toggle_mode(self):
        """
        Change le theme de l'application (sombre ou clair)
        """
        if self.dark:
            ctk.set_appearance_mode("light")
            self.dark = False
        else:
            ctk.set_appearance_mode("dark")
            self.dark = True
            
    def visualize_heat(self):
        """
        Visualiser la heat map
        """
        img_path = "map.png"
        df = self.raw_df.copy()
        df = convert_df(df)
        df = add_extreme(df, self.dataname)
        heatmap(df, self.dataname, img_path)
        
        on = Image.open("map_generated.png")
        map_image = ctk.CTkImage(dark_image=on, light_image=on, size=(852,402))
        self.map_label.configure(image=map_image)
        self.map_label.image = map_image
    
    def optionmenu_callback(self, choice):
        "permet de garder en mémoire le choix de l'optionMenu"
        self.dataname = choice
        
    def modelmenu_callback(self, choice):
        "permet de garder en mémoire le choix de l'optionMenu"
        self.modelline = int(choice)
        
    def predict(self, method="knn"):
        if(self.entry_file == ""):
            messagebox.showerror("Erreur", "Pas de fichier d'entrée selectionné!")
            return
        
        nb_line = self.modelline
        
        model = {'name': [], 'plk': [], 'predict': []}
        data = {'name': [], 'x':[], 'y':[]}
        
        df_train, df_test = prepare_data('data/train.csv', self.entry_file)
        df_test = pivot_dataset(df_test)
        Fingerprint_to_test = df_test.drop(['x', 'y'], axis=1).to_numpy()[nb_line]
        
        model = {'name': [], 'plk': [], 'predict': []}
        x, y = convert_position(df_test['x'][nb_line], df_test['y'][nb_line])
        data = {'name': ["real"], 'x':[x], 'y':[y]}
        
        
        if method in ["knn", "all"]:
            model['name'].append('knn')
            model['plk'].append('model/KNN_model.pkl')
            model['predict'].append(predict_knn)
        
        if method in ["tree", "all"]:
            model['name'].append('tree')
            model['plk'].append('model/tree_model.pkl')
            model['predict'].append(predict_tree)
        
        if method in ["m3", "all"]:
            model['name'].append('rf')
            model['plk'].append('model/RF_model.pkl')
            model['predict'].append(predict_rf)
        
        for name, plk, predict in zip(model['name'], model['plk'], model['predict']) :
            model = load_model(plk)
            pred = predict(model, [Fingerprint_to_test]).squeeze()
            x, y = convert_position(pred[0], pred[1])
            data['name'].append(name)
            data['x'].append(x)
            data['y'].append(y)
        
        df = pd.DataFrame(data)
        plot_dataframe(df, "map.png")
        
        on = Image.open("map_generated.png")
        map_image = ctk.CTkImage(dark_image=on, light_image=on, size=(852,402))
        self.map_label.configure(image=map_image)
        self.map_label.image = map_image
        
    def browse_file(self, mode):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            if mode == "raw":
                self.raw_file = file_path
                self.raw_file_name.set(os.path.basename(file_path))
                df = pd.read_csv(file_path)
                self.raw_df = df
                values = [col for col in df.columns if col not in ['x', 'y']]
                self.data_menu.configure(values = values)   
                self.data_menu.set("Sélectionner la donnée")
            elif mode == "entry":
                self.entry_file = file_path
                self.entry_file_name.set(os.path.basename(file_path))
                df_train, df_test = prepare_data('data/train.csv', self.entry_file)
                df_test = pivot_dataset(df_test)
                values = [str(index) for index in df_test.index]
                self.model_menu.configure(values= values)
                self.model_menu.set("Sélectionner la donnée")
                

ctk.set_appearance_mode("dark")     
root = ctk.CTk()
my_app = MyApp(root)
root.mainloop()
