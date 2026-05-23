import pandas as pd

def leer_csv(rutaArchivo) :

    #Todas las columnas : date,Usage_kWh,Lagging_Current_Reactive.Power_kVarh,Leading_Current_Reactive_Power_kVarh,CO2(tCO2),Lagging_Current_Power_Factor,Leading_Current_Power_Factor,NSM,WeekStatus,Day_of_week,Load_Type
    columnasEsperadas = ["date",
                         "Usage_kWh",
                         "Lagging_Current_Reactive.Power_kVarh",
                         "Leading_Current_Reactive_Power_kVarh",
                         "CO2(tCO2)",
                         "Lagging_Current_Power_Factor",
                         "Leading_Current_Power_Factor",
                         "NSM",
                         "WeekStatus",
                         "Day_of_week",
                         "Load_Type"]

    dataFrameCrudo = pd.read_csv(rutaArchivo)

    validarSchema(dataFrameCrudo, columnasEsperadas)

    dataFrame = parseDates(dataFrameCrudo, "date")

    dataFrame = dataFrame.sort_values("date").reset_index(drop = True)

    errores = dataFrame.isnull().sum().sum()

    if(errores > 0) :
        print(f"hay {errores} cantidad de nulos")
        return 
    
    timestampsDup = dataFrame["date"].duplicated().sum()

    if(timestampsDup > 0) :
        print(f"hay {timestampsDup} dates duplicados")

    return dataFrame

def validarSchema(dataFrameCrudo, columnasEsperadas) :
    set_actual = set(dataFrameCrudo.columns)
    set_esperado = set(columnasEsperadas)

    if set_actual != set_esperado :

        faltan = set_esperado - set_actual
        sobran = set_actual - set_esperado

        raise ValueError(
            f"[Loader] Mira, aquí hay información útil:\n "
            f"[Loader] los que hay: {set_actual}\n "
            f"[Loader] Mmmm al parecer hay un error con las columnas esperadas y lo que lees\n "
            f"[Loader] Faltan (set_esperado - set_actual) : {faltan}\n "
            f"[Loader] Sobran (set_actual - set_esperado) : {sobran}"
        )
    
    return 

def parseDates(dataFrameCrudo, colDate = "date") :
    try:
        dfCopia = dataFrameCrudo.copy()

        dfCopia[colDate] = pd.to_datetime(dfCopia[colDate], dayfirst = True, errors = "raise")

        return dfCopia

    except Exception as e :
        raise ValueError(
            f"[Loader] Error al convertir fechas"
            f"[Loader] mensaje del error :\n{e}"
            ) from e
    