import json
import numpy as np

from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_test : np.ndarray,
                 y_pred_a : np.ndarray,
                 y_pred_b : np.ndarray,
                 nombre_a : str = "Modelo A",
                 nombre_b : str = "Modelo B",
                 alpha : float = 0.05
                 ) -> dict:
    a_correcto = (y_pred_a == y_test)
    b_correcto = (y_pred_b == y_test)

    ambos_ok = int(np.sum(a_correcto & b_correcto))
    solo_a  = int(np.sum(a_correcto & ~b_correcto))
    solo_b = int(np.sum(~a_correcto & b_correcto))
    ambos_mal = int(np.sum(~a_correcto & ~b_correcto))

    tabla = np.array([
        [ambos_ok, solo_a],
        [solo_b, ambos_mal]
    ])

    usar_exacto = (solo_a + solo_b) < 25
    resultado = mcnemar(tabla, exact = usar_exacto, correction = True)

    rechazar_h0 = resultado.pvalue < alpha
    interpretacion = ""

    if rechazar_h0 :
        ganador = nombre_a if solo_a > solo_b else nombre_b
        interpretacion = (f"Se rechaza H0 con p: {resultado.pvalue:.4f} < alpha: {alpha}\n "
                          f"El {ganador} es significativamente mejor")
        
    else:
        ganador = "Empate, sin diferencia significativa"
        interpretacion = (f"No se rechaza H0 con p: {resultado.pvalue:.4f} >= aplpha: {alpha}\n "
                          f"Los modelos son signifcativamente indistinguibles")
        
    return {"modelo_a" : nombre_a,
            "modelo_b" : nombre_b,
            "tabla_contingencia" : tabla.tolist(),
            "ambos_correctos" : ambos_ok,
            "solo_a" : solo_a,
            "solo_b" : solo_b,
            "ambos_mal" : ambos_mal,
            "estadistico" : float(resultado.statistic),
            "resultado" : float(resultado.pvalue),
            "alpha" : alpha,
            "metodo" : "exacto" if usar_exacto else "chi cuadrado con ajuste",
            "rechazar_H0" : bool(rechazar_h0),
            "ganador" : ganador,
            "interpretacion" : interpretacion
            }

def imprimir_resultado_mcnemar(resultado: dict) -> None:

    mod_a = resultado["modelo_a"]
    mod_b = resultado["modelo_b"]

    print("=" * 60)
    print(f"TEST DE MCNEMAR: {mod_a} vs {mod_b}".center(60))
    print("=" * 60)

    ok_a_ok_b = resultado["ambos_correctos"]
    ok_a_fail_b = resultado["solo_a"]
    fail_a_ok_b = resultado["solo_b"]
    fail_a_fail_b = resultado["ambos_mal"]

    lbl_ok_b = f"{mod_b} acierta"
    lbl_fail_b = f"{mod_b} falla"
    lbl_ok_a = f"{mod_a} acierta"
    lbl_fail_a = f"{mod_a} falla"

    print("Tabla de contingencia:")
    print(f"{'':<20} {lbl_ok_b:<15} {lbl_fail_b:<15}")
    print(f"{lbl_ok_a:<20} {ok_a_ok_b:<15} {ok_a_fail_b:<15}")
    print(f"{lbl_fail_a:<20} {fail_a_ok_b:<15} {fail_a_fail_b:<15}\n")

    print(f"Estadístico: {resultado['estadistico']:.4f}")
    print(f"P-valor:     {resultado['resultado']:.4f}")
    print(f"Alpha:       {resultado['alpha']}")
    print(f"Método:      {resultado['metodo']}")
    print()

    p_val = resultado["resultado"]
    alpha = resultado["alpha"]

    if resultado["rechazar_H0"]:
        conclusion = f"Se rechaza H₀ con p={p_val:.4f} < α={alpha}."
    else:
        conclusion = f"No se rechaza H₀ con p={p_val:.4f} >= α={alpha}."

    print(f"Resultado: {conclusion}")
    print(f"           {resultado['interpretacion']}")
    print("=" * 60)

def guardar_resultado_mcnemar(resultado : dict, ruta_archivo : Path) :

    with open(ruta_archivo, "w", encoding = "utf-8") as f :
        json.dump(resultado, f, indent = 2)

    print(f"[Statistical] resultado McNemar guardado en {ruta_archivo}")

    return ruta_archivo
