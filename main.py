import src.data.loader as loader
import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
from src.config import VALORES_FUGA, MESES_ENTRENAMIENTO, MESES_EVALUACION

import json
from pathlib import Path


def main():
    
    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))
    df, mapeo_target = preprocessor.preprocesar(df)
    print(f"Shape: {df.shape}")              # esperado: (35040, 9)
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Mapeo target: {mapeo_target}")
    print(df.head())
    print(f"Distribución target:\n{df['Load_Type'].value_counts(normalize=True).round(3)}")

    train, evaluar, test = splitter.dividir_temporal(df, "date", MESES_ENTRENAMIENTO, MESES_EVALUACION)
    reporte = splitter.verificar_split(train, evaluar, test, "Load_Type", "date")

    print(json.dumps(reporte, indent=2, default=str))

if __name__ == "__main__":
    main()