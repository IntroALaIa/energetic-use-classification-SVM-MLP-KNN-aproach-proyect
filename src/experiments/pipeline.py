import src.experiments.preparacion as preparacion
import src.experiments.entrenamiento as entrenamiento
import src.experiments.evaluacion as evaluacion
import src.experiments.comparacion as comparacion

from src.config import (SEMILLA)

import pandas as pd

def ejecutar_experimento_completo(df : pd.DataFrame,
                                  meses_train : int,
                                  meses_val : int,
                                  semilla : int,
                                  alpha : float,
                                  sufijo : str = "") -> dict :
    
    print(f"\n{'='*7} Iniciando proyecto {sufijo or 'principal'} un segundo {'='*60} ")

    datos = preparacion.preparar_datos(df, meses_train, meses_val)
    modelos = entrenamiento.entrenar_modelos(datos, semilla)
    metricas_todos, predicciones_por_modelo = evaluacion.evaluar_modelos(modelos, datos, sufijo)
    comparaciones = comparacion.comparar_modelos_estadisticamente(modelos, datos, predicciones_por_modelo, alpha, sufijo)

    return {
        "datos" : datos,
        "modelos" : modelos,
        "metricas_todos" : metricas_todos,
        "comparaciones" : comparaciones
    }
