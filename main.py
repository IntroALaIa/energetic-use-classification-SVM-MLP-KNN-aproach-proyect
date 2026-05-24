import src.data.loader as loader
import src.experiments.pipeline as pipeline

from pathlib import Path
from src.config import (DATA_RAW,
                        MESES_ENTRENAMIENTO,
                        MESES_EVALUACION,
                        SEMILLA,
                        ALPHA_SIGNIFICANCIA,
                        fijar_semillas
                        )

def main() :
    
    fijar_semillas()

    df = loader.leer_csv(DATA_RAW)

    resultados = pipeline.ejecutar_experimento_completo(df, 
                                                        MESES_ENTRENAMIENTO, 
                                                        MESES_EVALUACION, SEMILLA, 
                                                        ALPHA_SIGNIFICANCIA, 
                                                        "principal")
    return

if __name__ == "__main__" :


    main()