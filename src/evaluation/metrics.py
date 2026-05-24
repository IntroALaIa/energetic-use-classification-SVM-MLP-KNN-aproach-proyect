import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix
                             )

def calcular_metricas(y_true : np.ndarray,
                      y_pred : np.ndarray,
                      y_proba : np.ndarray,
                      nombres_clases : list[str]) -> dict :
    metricas_generales = {
        "exactitud" : float(accuracy_score(y_true, y_pred)),
        "exactitud_balanceada" : float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro" : float(f1_score(y_true, y_pred, average = "macro")),
        "f1_pesado" : float(f1_score(y_true, y_pred, average = "weighted")),
        "roc_auc_macro" : float(roc_auc_score(y_true, y_proba, multi_class = "ovr", average = "macro"))
    }

    precs = precision_score(y_true, y_pred, average=None)
    recs = recall_score(y_true, y_pred, average=None)
    f1s = f1_score(y_true, y_pred, average=None)

    
    soporte = np.bincount(y_true)

    
    metricas_por_clase = {
        nombre: {
            "precision": float(precs[i]),
            "recall": float(recs[i]),
            "f1": float(f1s[i]),
            "support": int(soporte[i]),
        }
        for i, nombre in enumerate(nombres_clases)
    }

    mc = confusion_matrix(y_true, y_pred).tolist()
    
    return {
            "metricas_generales" : metricas_generales,
            "metricas_por_clase" : metricas_por_clase,
            "matriz_confusion" : mc
            }

def guardar_metricas(metricas : dict, nombre_modelo : str, ruta_archivo : Path) -> Path :
    archivo = ruta_archivo / f"{nombre_modelo}_metricas.json"

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent = 2)

    print(f"[Metrics] Métrica guardada en {archivo}")

    return archivo

def imprimir_resumen_comparativo(metricas_por_modelo : dict[str, dict]) -> None :
    
    print(f"\n{'Modelo':<12} {'F1 macro':>10} {'Acc':>10} {'Bal Acc':>10} {'ROC-AUC':>10}")
    print("-" * 56)

    for nombre, metricas in metricas_por_modelo.items():
        
        generales = metricas["metricas_generales"]
        f1 = generales.get("f1_macro", 0.0)
        acc = generales.get("exactitud", 0.0)
        bal_acc = generales.get("exactitud_balanceada", 0.0)
        roc_auc = generales.get("roc_auc_macro", 0.0)

        print(f"{nombre:<12} {f1:>10.4f} {acc:>10.2f} {bal_acc:>10.2f} {roc_auc:>10.2f}")