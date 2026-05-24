import src.evaluation.metrics as metrics
import src.evaluation.visualization as visualization
from src.config import (
                        METRICS,
                        FIGURES
                        )

def evaluar_modelos(modelos :dict,
                    datos: dict,
                    sufijo_experimento : str = "") :
    nombre_clases = list(datos["mapeo_target"].keys())

    metricas_todos = {}
    probas_por_modelo = {}
    predicciones_por_modelo = {}

    for nombre, modelo in [("Dummy", datos["DUMMY"]),("LogReg", datos["LogReg"]),("SVM", datos["SVM"]), ("MLP", datos["MLP"])]:

        print(f"{nombre}========================================")

        objetivo_predict = modelo.predict(datos["caracteristicas_test"])
        objetivo_proba = modelo.predict_proba(datos["caracteristicas_test"])

        probas_por_modelo[nombre] = (datos["objetivo_test"].values, objetivo_proba)
        predicciones_por_modelo[nombre] = objetivo_predict


        metricas = metrics.calcular_metricas(datos["objetivo_test"].values, objetivo_predict, objetivo_proba, nombre_clases)
        metrics.guardar_metricas(metricas, nombre, METRICS)
        metricas_todos[nombre] = metricas

        visualization.graficar_matriz_confusion(datos["objetivo_test"].values,
                                                objetivo_predict,
                                                nombre_clases,
                                                f"matriz confusión - {nombre}",
                                                FIGURES / f"matriz_confusion_{nombre}.png",
                                                True)
        visualization.graficar_curva_roc_multiclase(datos["objetivo_test"],
                                                    objetivo_proba,
                                                    nombre_clases,
                                                    f"Curva ROC OvR - {nombre}",
                                                    FIGURES / f"roc_{nombre}.png"
                                                    )
    
    visualization.graficar_roc_comparativa(probas_por_modelo, nombre_clases, FIGURES / "roc_comparativa.png")
    visualization.graficar_curvas_aprendizaje(datos["MLP"].history, FIGURES / "curvas_aprendizaje_mlp.png")

    metrics.imprimir_resumen_comparativo(metricas_todos)

    return metricas_todos, predicciones_por_modelo
    
