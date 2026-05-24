VALORES_FUGA = ["Usage_kWh", "CO2(tCO2)"]
SEMILLA = 42

MESES_ENTRENAMIENTO = 9
MESES_EVALUACION = 1


SVM_GRIDSEARCH_SUBSAMPLE = 8000
SVM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf"],
}