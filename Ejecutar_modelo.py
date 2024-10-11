#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:00:57 2024

@author: javiermoreno
"""

import xgboost as xgb
import pandas as pd

workdir = '/Users/javiermoreno/Documents/Agrosavia_Capsicum_2023/Entregables/Modelos/Modelo_XGBoost/'
muestra = pd.read_excel(workdir + 'Muestra_input.xlsx')
variables = ['Color de la mancha de la corola_2', 
             'Color de la mancha de la corola_3', 
             'Número de flores por axila_1', 
             'Color de la mancha de la corola_4', 
             'Persistencia del pedicelo en el tallo_7', 
             'Color de la corola_1', 
             'Color de la mancha de la corola_0', 
             'Color de la mancha de la corola_1', 'Exserción del estigma_3', 
             'Color de la corola_3', 'Ancho de fruto', 'Color del filamento_7', 
             'Longitud hoja cotidoneidal (mm)', 'Pubescencia del Tallo_7', 
             'Peso fruto', 'Color del fruto en estado inmaduro_6', 
             'Arrugamiento transversal del fruto_5', 'Pubescencia del Tallo_3', 
             'Longitud de la corola_3', 'Color de la corola_4'] 


resultados = muestra.copy()
muestra = muestra.reindex(columns=variables)

xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model(workdir + 'model_xgboost.json')


muestra["Ancho de fruto"] = (muestra["Ancho de fruto"] - 2.054929)/1.342260
muestra["Longitud hoja cotidoneidal (mm)"] = (muestra["Longitud hoja cotidoneidal (mm)"] - 31.028143)/11.826836
muestra["Peso fruto"] = (muestra["Peso fruto"] - 12.783778)/18.310342


predicted = xgb_cl.predict(muestra)

especies = {
    0: "C. pubescens",
    1: "C. frutescens",
    2: "C. annuum",
    3: "C. baccatum",
    4: "C. chinense",
    5: "C. abbreviatum",
    6: "C. flexuosum"
}

resultados["Prediccion"] = pd.Series(predicted).map(especies)

resultados.to_excel(workdir+'Muestra_output.xlsx', index=False)
