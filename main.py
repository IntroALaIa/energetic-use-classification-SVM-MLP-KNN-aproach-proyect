import src.data.loader as loader
import src.data.preprocessor as preprocessor
from src.config import VALORES_FUGA

from pathlib import Path


def main():
    
    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))
    df, mapeo_target = preprocessor.preprocesar(df)
    print(f"Shape: {df.shape}")              # esperado: (35040, 9)
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Mapeo target: {mapeo_target}")
    print(df.head())
    print(f"Distribución target:\n{df['Load_Type'].value_counts(normalize=True).round(3)}")



if __name__ == "__main__":
    main()