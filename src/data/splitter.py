from sklearn.model_selection import train_test_split
import pandas as pd

#divide con respecto a una linea de tiempo, no como datos regados, y saca el train, evaluar, y test
def dividir_temporal(data_frame : pd.DataFrame, col_fecha, meses_train, meses_val) :
    data_frame_copia = data_frame.copy()

    data_frame_copia = data_frame_copia.sort_values(col_fecha).reset_index(drop = True)

    inicio = data_frame_copia[col_fecha].min()
    fin_train = inicio + pd.DateOffset(months = meses_train)
    fin_evaluar = fin_train + pd.DateOffset(months = meses_val)

    train = data_frame_copia[data_frame_copia[col_fecha] < fin_train].reset_index(drop = True)
    evaluar = data_frame_copia[(data_frame_copia[col_fecha] >= fin_train) & (data_frame_copia[col_fecha] < fin_evaluar)].reset_index(drop = True)
    test = data_frame_copia[data_frame_copia[col_fecha] >= fin_evaluar].reset_index(drop = True)
                            
    return train, evaluar, test

def submuestreo_estratificado(data_frame, col_objetivo, n_muestras, semilla = 42):

    data_frame_copia = data_frame.copy()

    if(n_muestras > len(data_frame)) :
        return data_frame_copia
    
    #puse todo, no entendí bien a que se referia con descartar
    submuestreo, _ = train_test_split(data_frame_copia,
                                      train_size = n_muestras, 
                                      stratify = data_frame_copia[col_objetivo], 
                                      random_state = semilla)
    
    submuestreo = submuestreo.reset_index(drop = True)

    return submuestreo

def verificar_split(train, evaluar, test, col_objetivo, col_fecha) :

    if train[col_fecha].max() >= evaluar[col_fecha].min() :
        raise ValueError(
            f"[Splitter] El último dato de entrenamiento se sobrepone con el primero de evaluar"
        )
    
    elif evaluar[col_fecha].max() >= test[col_fecha].min() :
        raise ValueError(
            f"[Splitter] El último dato de evaluar se sobrepone con el primero de test"
        )
    
    n_clases_train = train[col_objetivo].nunique()
    n_clases_evaluar = evaluar[col_objetivo].nunique()
    n_clases_test = test[col_objetivo].nunique()

    if(n_clases_train < 3 or n_clases_evaluar < 3 or n_clases_test < 3) :
        raise ValueError(
            "[Splitter] Aparentemente hubo un error en la distribución\n"
            f"[Splitter] n clases de train: {n_clases_train}\n"
            f"[Splitter] n clases de evaluar : {n_clases_evaluar}\n"
            f"[Splitter] n clases de test : {n_clases_test}\n"
        )
    
    diccionario = {
        "train" : generar_diccionario(train, col_fecha, col_objetivo),
        "evaluar" : generar_diccionario(evaluar, col_fecha, col_objetivo),
        "test" : generar_diccionario(test, col_fecha, col_objetivo)
    }

    return diccionario

def generar_diccionario(data_frame, col_fecha, col_objetivo) :

    tamanio = len(data_frame)
    rango_fechas = data_frame[col_fecha].min(), data_frame[col_fecha].max()
    distribucion_clases = data_frame[col_objetivo].value_counts(normalize = True).round(3).to_dict()

    return  {"tamanio" :tamanio, "rango_fechas" : rango_fechas, "distribucion_clases" : distribucion_clases}
    
