import src.data.loader as loader
import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
import src.features.engineering as engineering
import src.evaluation.metrics as metrics
import src.evaluation.visualization as visualization
import src.evaluation.statistical as statistical

from src.models.base import BaseModel
from src.models.svm_model import SVMmodel
from src.models.mlp_model import MLPmodel
from src.config import (
                        VALORES_FUGA, 
                        MESES_ENTRENAMIENTO, 
                        MESES_EVALUACION,
                        SEMILLA,
                        SVM_GRID,
                        SVM_GRIDSEARCH_SUBSAMPLE,
                        MODELS,
                        FIGURES,
                        METRICS,
                        ALPHA_SIGNIFICANCIA
                        )

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def main():
    

    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))

    df, mapeo_target = preprocessor.preprocesar(df, VALORES_FUGA)

    df = engineering.aplicar_caracteristicas_ciclicas(df)

    train, evaluar, test = splitter.dividir_temporal(df, "date", MESES_ENTRENAMIENTO, MESES_EVALUACION)
    reporte = splitter.verificar_split(train, evaluar, test, "Load_Type", "date")

    print(f"Reporte de los splits:\n{json.dumps(reporte, indent = 2, default=str)}")

    #Separar los sets em caracteristicas y objetivos
    objetivo_train = train["Load_Type"]
    caractaristicas_train = train.drop(columns = ["date", "Load_Type"])

    objetivo_evaluar = evaluar["Load_Type"]
    caracteristicas_evaluar = evaluar.drop(columns = ["date", "Load_Type"])

    objetivo_test = test["Load_Type"]
    caracteristicas_test = test.drop(columns = ["date", "Load_Type"])

    scaler = StandardScaler()
    caracteristicas_train_escalado = scaler.fit_transform(caractaristicas_train)
    caracteristicas_evaluar_escalado = scaler.transform(caracteristicas_evaluar)
    caracteristicas_test_escalado = scaler.transform(caracteristicas_test)

    svm_model : BaseModel = SVMmodel({"SVM_GRID" : SVM_GRID, 
                                       "tam_subsampleo" : SVM_GRIDSEARCH_SUBSAMPLE, 
                                       "semilla" : SEMILLA})
    
    svm_model.fit(caracteristicas_train_escalado, objetivo_train, "Load_Type")

    objetivo_predict = svm_model.predict(caracteristicas_test_escalado)

    mlp_model : BaseModel = MLPmodel(n_features = caracteristicas_train_escalado.shape[1])
    mlp_model.fit(caracteristicas_train_escalado, objetivo_train,
                  caracteristicas_evaluar_escalado, objetivo_evaluar)
    mlp_model.save( MODELS / "mlp_model.keras")

    dummy = DummyClassifier(strategy='stratified', random_state=SEMILLA)
    dummy.fit(caracteristicas_train_escalado, objetivo_train)

    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEMILLA)
    logreg.fit(caracteristicas_train_escalado, objetivo_train)

    metricas_todos = {}
    probas_por_modelo = {}
    predicciones_por_modelo = {}

    nombres = list(mapeo_target.keys())

    for nombre, modelo in [("Dummy", dummy),("LogReg", logreg),("SVM", svm_model), ("MLP", mlp_model)]:

        print(f"{nombre}========================================")

        objetivo_predict = modelo.predict(caracteristicas_test_escalado)
        objetivo_proba = modelo.predict_proba(caracteristicas_test_escalado)

        probas_por_modelo[nombre] = (objetivo_test.values, objetivo_proba)
        predicciones_por_modelo[nombre] = objetivo_predict


        metricas = metrics.calcular_metricas(objetivo_test.values, objetivo_predict, objetivo_proba, nombres)
        metrics.guardar_metricas(metricas, nombre, METRICS)
        metricas_todos[nombre] = metricas

        visualization.graficar_matriz_confusion(objetivo_test.values,
                                                objetivo_predict,
                                                nombres,
                                                f"matriz confusión - {nombre}",
                                                FIGURES / f"matriz_confusion_{nombre}.png",
                                                True)
        visualization.graficar_curva_roc_multiclase(objetivo_test,
                                                    objetivo_proba,
                                                    nombres,
                                                    f"Curva ROC OvR - {nombre}",
                                                    FIGURES / f"roc_{nombre}.png"
                                                    )
    
    visualization.graficar_roc_comparativa(probas_por_modelo, nombres, FIGURES / "roc_comparativa.png")
    visualization.graficar_curvas_aprendizaje(mlp_model.history, FIGURES / "curvas_aprendizaje_mlp.png")

    resultado_svm_vs_mp = statistical.mcnemar_test(objetivo_test,
                                                   predicciones_por_modelo["SVM"],
                                                   predicciones_por_modelo["MLP"],
                                                   nombre_a = "SVM",
                                                   nombre_b = "MLP",
                                                   alpha = ALPHA_SIGNIFICANCIA)
    statistical.imprimir_resultado_mcnemar(resultado_svm_vs_mp)
    statistical.guardar_resultado_mcnemar(resultado_svm_vs_mp, METRICS / "svm_vs_mlp_mcnemar.json")

    resultado_svm_vs_logreg = statistical.mcnemar_test(objetivo_test,
                                                   predicciones_por_modelo["SVM"],
                                                   predicciones_por_modelo["LogReg"],
                                                   nombre_a = "SVM",
                                                   nombre_b = "LogReg",
                                                   alpha = ALPHA_SIGNIFICANCIA)
    statistical.imprimir_resultado_mcnemar(resultado_svm_vs_logreg)
    statistical.guardar_resultado_mcnemar(resultado_svm_vs_logreg, METRICS / "svm_vs_logReg_mcnemar.json")

    resultado_mlp_vs_logreg = statistical.mcnemar_test(objetivo_test,
                                                   predicciones_por_modelo["MLP"],
                                                   predicciones_por_modelo["LogReg"],
                                                   nombre_a = "MLP",
                                                   nombre_b = "LogReg",
                                                   alpha = ALPHA_SIGNIFICANCIA)
    statistical.imprimir_resultado_mcnemar(resultado_mlp_vs_logreg)
    statistical.guardar_resultado_mcnemar(resultado_mlp_vs_logreg, METRICS / "logReg_vs_mlp_mcnemar.json")



    metrics.imprimir_resumen_comparativo(metricas_todos)
    # Concatene los tres splits con una columna que diga cuál
    train["split"] = "train"
    evaluar["split"] = "val"
    test["split"] = "test"
    todo = pd.concat([train, evaluar, test])
    todo["mes"] = todo["date"].dt.month

    # Distribución del target por mes
    print(todo.groupby("mes")["Load_Type"].value_counts(normalize=True).unstack())

    # Distribución de features clave por split
    features_clave = ["Lagging_Current_Reactive.Power_kVarh", "Lagging_Current_Power_Factor"]
    for f in features_clave:
        todo.boxplot(column=f, by="split")
        plt.title(f)
        plt.show()
        plt.savefig(FIGURES / f"{f}_image.png")


if __name__ == "__main__":
    main()