import src.data.loader as loader


def main():

    csvOriginal = "Steel_industry_data.csv"
    csvProbarQueFunciona = "prueba.csv"

    dataFrame = loader.leerCSV(csvProbarQueFunciona, "processed")

    print(f"Dataframe:\n {dataFrame}")

    return



if __name__ == "__main__":
    main()