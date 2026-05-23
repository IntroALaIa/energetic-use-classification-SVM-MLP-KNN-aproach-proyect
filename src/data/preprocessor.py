from src.config import VALORES_FUGA
import pandas as pd

MAPEO_WEEK_STATUS = { "Weekday" : 1, "Weekend" : 0 }
MAPEO_DIA_SEMANA = { 
                    "Sunday": 0,
                    "Monday" : 1,
                    "Tuesday" : 2,
                    "Wednesday" : 3, 
                    "Thursday" : 4,
                    "Friday" : 5,
                    "Saturday" : 6
                    }
MAPEO_LOAD_TYPE = { "Light_Load" : 0, "Medium_Load" : 1, "Maximum_Load" : 2 }

def eliminar_features_fuga(data_frame, columnas_eliminar) :

    data_frame_copia = data_frame.copy()

    buscar = set(columnas_eliminar)
    encontrados = [x for x in data_frame_copia.columns if x in buscar]

    if(len(encontrados) != len(columnas_eliminar)) :
        raise ValueError(
            f"[Preprocessor] Las columnas difieren, las que se quieren eliminar no están\n"
            f"[Preprocessor] columnas actuales: {data_frame_copia.columns}\n"
            f"[Preprocessor] columnas para eliminar : {columnas_eliminar}"
        )
    
    return data_frame_copia.drop(columns = columnas_eliminar)

    
def codificar_week_status(data_frame, col = "WeekStatus") :
    data_frame_copia = data_frame.copy()

    valores_unicos = set(data_frame_copia[col].dropna().unique())
    esperados = set(MAPEO_WEEK_STATUS.keys())

    validacion_valores_unicos(valores_unicos, esperados, col)

    data_frame_copia[col] = data_frame_copia[col].map(MAPEO_WEEK_STATUS).astype(int)

    return data_frame_copia

def codificar_dia_semana(data_frame, col = "Day_of_week") :
    data_frame_copia = data_frame.copy()

    valores_unicos = set(data_frame_copia[col].dropna().unique())
    esperados = set(MAPEO_DIA_SEMANA.keys())

    validacion_valores_unicos(valores_unicos, esperados, col)

    data_frame_copia[col] = data_frame_copia[col].map(MAPEO_DIA_SEMANA).astype(int)

    return data_frame_copia

def codificar_load_status(data_frame, col = "Load_Type") :
    data_frame_copia = data_frame.copy()

    valores_unicos = set(data_frame_copia[col].dropna().unique())
    esperados = set(MAPEO_LOAD_TYPE.keys())

    validacion_valores_unicos(valores_unicos, esperados, col)

    data_frame_copia[col] = data_frame_copia[col].map(MAPEO_LOAD_TYPE).astype(int)

    return data_frame_copia, MAPEO_LOAD_TYPE

def preprocesar(data_frame) :
    df_nuevo = eliminar_features_fuga(data_frame, VALORES_FUGA)

    df_nuevo = codificar_week_status(df_nuevo)
    df_nuevo = codificar_dia_semana(df_nuevo)
    df_nuevo, mapeo_target = codificar_load_status(df_nuevo)

    return df_nuevo, mapeo_target

def validacion_valores_unicos(valores_unicos, esperados, col) :
    if valores_unicos != esperados :
        raise ValueError(
        f"[Preprocessor] valores inesperados en {col}\n"
        f"[Preprocessor] encontrados : {valores_unicos}"
        f"[Preprocessor] esperados : {esperados}"
    )