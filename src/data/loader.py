import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent.parent
sys.path.append(str(RAIZ))

import pandas as pd

def leerCSV(nombreCSV, rutaCarpeta) :

    rutaArchivo = RAIZ / "data" / rutaCarpeta / nombreCSV

    #Todas las columnas : date,Usage_kWh,Lagging_Current_Reactive.Power_kVarh,Leading_Current_Reactive_Power_kVarh,CO2(tCO2),Lagging_Current_Power_Factor,Leading_Current_Power_Factor,NSM,WeekStatus,Day_of_week,Load_Type

    columnasUtilizar = ["Lagging_Current_Reactive.Power_kVarh",
                        "Leading_Current_Reactive_Power_kVarh", 
                        "Lagging_Current_Power_Factor",
                        "Leading_Current_Power_Factor",
                        "NSM",
                        "WeekStatus",
                        "Day_of_week",
                        "Load_Type"
]
    dataFrame = pd.read_csv(rutaArchivo, usecols = columnasUtilizar)

    return dataFrame

