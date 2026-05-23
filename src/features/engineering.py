import numpy as np
import pandas as pd

PERIODO_NSM = 86400
PERIODO_DIA_SEMANA = 7

def codificar_ciclica(data_frame, col, periodo, prefijo) :
    data_frame_copia = data_frame.copy()

    angulo = 2 * np.pi * data_frame_copia[col] / periodo

    data_frame_copia[f"{prefijo}_sin"] = np.sin(angulo)
    data_frame_copia[f"{prefijo}_cos"] = np.cos(angulo)

    data_frame_copia = data_frame_copia.drop(columns = [col])
    
    return data_frame_copia

def aplicar_caracteristicas_ciclicas(data_frame) :
    data_frame_copia = data_frame.copy()

    data_frame_copia = codificar_ciclica(data_frame_copia, "NSM", PERIODO_NSM, "nsm")
    data_frame_copia = codificar_ciclica(data_frame_copia, "Day_of_week", PERIODO_DIA_SEMANA, "dow")

    return data_frame_copia