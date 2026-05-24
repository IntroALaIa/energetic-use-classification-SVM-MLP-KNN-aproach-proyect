import src.evaluation.statistical as statistical
from src.config import METRICS

def comparar_modelos_estadisticamente(modelos : dict,
                                      datos : dict,
                                      predicciones : dict,
                                      alpha : float,
                                      sufijo_experimento : str = ""
                                      ) -> dict :
    pares = [
        ("SVM", "MLP"),
        ("SVM", "LogReg"),
        ("MLP", "LogReg")
    ]

    resultados = {}

    for nombre_a, nombre_b in pares :
        resultado = statistical.mcnemar_test(datos["objetivo_test"].values,
                                             predicciones[nombre_a],
                                             predicciones[nombre_b],
                                             nombre_a,
                                             nombre_b,
                                             alpha)
        statistical.imprimir_resultado_mcnemar(resultado)
        
        clave = f"{nombre_a}_vs_{nombre_b}"
        statistical.guardar_resultado_mcnemar(resultado, METRICS / f"mc_nemar_{clave}_{sufijo_experimento}")

        resultados[clave] = resultado

    return resultados