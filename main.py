from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Any, Dict, AnyStr, List, Union
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from joblib import load
import pickle

app = FastAPI()


tablas_enc = load('tables.joblib')
clf1 = load('clf1.joblib')
clf2 = load('clf2.joblib')

class Item(BaseModel):
  Edadcat: str
  sexocat: str
  grupo_sanguineocat: str
  Pesocat: str
  Tallacat: str
  Tempcat: str
  F_Ccat: str
  F_Rcat: str
  TEST_DE_FINDRISKcat: str
  TASA_DE_FILTRACION_GLOMERULARcat: str
  TEST_DE_FRAMINGHAMcat: str
  diagnostico_principalcat: str
  #diagnostico_secundariocat: str
  #diagnostico_terciariocat: str
  #cuarto_diagnosticocat: str
  T_a1cat:str
  T_a2cat: str



@app.post("/")
async def root(item: Item):
    dato_en=[]
    dato_en.append(int(item.Edadcat))
    dato_en.append(tablas_enc['sexo'].transform([[item.sexocat]])[0][0])
    dato_en.append(float(item.Tallacat))
    dato_en.append(tablas_enc['grupo_sanguineo'].fit_transform([[item.grupo_sanguineocat]])[0][0])
    dato_en.append(float(item.Pesocat))
    dato_en.append(float(item.Tempcat))
    dato_en.append(int(item.F_Ccat))#F_C
    dato_en.append(int(item.F_Rcat))#F_R
    dato_en.append(float(item.TEST_DE_FINDRISKcat))#Test de frindrisk
    dato_en.append(float(item.TASA_DE_FILTRACION_GLOMERULARcat))#Tasa de filtracion
    dato_en.append(float(item.TEST_DE_FRAMINGHAMcat))#Test de framingan ,porcentaje
    dato_en.append(tablas_enc['diagnostico_principal'].fit_transform([[item.diagnostico_principalcat]])[0][0])#diagnostico_principal
    #dato_en.append(tablas_enc['diagnostico_secundario'].fit_transform([[item.diagnostico_secundariocat]]))#diagnostico_secundario
    #dato_en.append(tablas_enc['diagnostico_terciario'].fit_transform([[item.diagnostico_terciariocat]]))#diagnostico_terciario
    #dato_en.append(tablas_enc['cuarto_diagnostico'].fit_transform([[item.cuarto_diagnosticocat]]))#cuarto_diagnostico
    dato_en.append(int(item.T_a1cat))#T_a1
    dato_en.append(int(item.T_a2cat))#T_a2
    prediccion= clf1.predict([dato_en])
    lists = prediccion.tolist()
    prediccion2= clf2.predict(lists)
    lists2 = prediccion2.tolist()
    respuesta3=0
    respuesta3={
        "cups_solictud":tablas_enc['cups_solictud'].inverse_transform([[prediccion[0][0]]])[0][0],
        "descripcion" :tablas_enc['descripcion'].inverse_transform([[prediccion[0][1]]])[0][0],
        "cantidad":tablas_enc['cantidad'].inverse_transform([[prediccion[0][2]]])[0][0],
        "prescripcion": tablas_enc['prescripcion'].inverse_transform([[prediccion[0][3]]])[0][0][0][0],
        "tipo": tablas_enc['tipo'].inverse_transform([[prediccion[0][4]]])[0][0],
        "codigo_medicamento":tablas_enc['codigo_medicamento'].inverse_transform([[prediccion2[0][0]]])[0][0],
        "descripcion_medicamento":tablas_enc['descripcion_medicamento'].inverse_transform([[prediccion2[0][1]]])[0][0],
        "cantidad_medicamento":tablas_enc['cantidad_medicamento'].inverse_transform([[prediccion2[0][2]]])[0][0],
        "prescripcion_medicamento":tablas_enc['prescripcion_medicamento'].inverse_transform([[prediccion2[0][3]]])[0][0],
        #"presentacion":tablas_enc['presentacion'].inverse_transform([[prediccion2[0][4]]])[0][0]
        }
    return respuesta3