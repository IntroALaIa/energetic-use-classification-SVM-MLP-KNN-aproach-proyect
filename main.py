import src.data.loader as loader
from pathlib import Path


def main():
    
    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))
    print(f"Shape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Rango fechas: {df['date'].min()} → {df['date'].max()}")
    print(df.head())


if __name__ == "__main__":
    main()