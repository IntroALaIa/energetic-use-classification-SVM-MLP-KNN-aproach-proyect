import pandas as pd

from sklearn.preprocessing import StandardScaler

import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
import src.features.engineering as engineering

from src.config import (VALORES_FUGA)

def preparar_datos(data_frame : pd.DataFrame,
                   meses_train : int,
                   meses_val : int,
                   col_fecha : str = "date",
                   col_objetivo : str = "Load_Type") -> dict:
    
    df, mapeo_target = preprocessor.preprocesar(data_frame, VALORES_FUGA)

    df = engineering.aplicar_caracteristicas_ciclicas(df)

    train, evaluar, test = splitter.dividir_temporal(df, col_fecha, meses_train, meses_val)
    reporte = splitter.verificar_split(train, evaluar, test, col_objetivo, col_fecha)

    #Separar los sets em caracteristicas y objetivos
    objetivo_train = train["Load_Type"]
    caractaristicas_train = train.drop(columns = ["date", "Load_Type"])

    objetivo_evaluar = evaluar["Load_Type"]
    caracteristicas_evaluar = evaluar.drop(columns = ["date", "Load_Type"])

    objetivo_test = test["Load_Type"]
    caracteristicas_test = test.drop(columns = ["date", "Load_Type"])

    scaler = StandardScaler()
    caracteristicas_train_escalado = scaler.fit_transform(caractaristicas_train)
    caracteristicas_evaluar_escalado = scaler.transform(caracteristicas_evaluar)
    caracteristicas_test_escalado = scaler.transform(caracteristicas_test)
    
    return {
        "objetivo_train" : objetivo_train,
        "caracteristicas_train" : caracteristicas_train_escalado,
        "objetivo_evaluar" : objetivo_evaluar,
        "caracteristicas_evaluar" : caracteristicas_evaluar_escalado,
        "objetivo_test" : objetivo_test,
        "caracteristicas_test" : caracteristicas_test_escalado,
        "mapeo_target" : mapeo_target,
        "reporte_split"  : reporte,
        "n_features" : caracteristicas_train_escalado.shape[1]
    }
