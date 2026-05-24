import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

sns.set_theme(style = "whitegrid", context = "notebook")
DPI = 150
FIGSIZE = (8, 6)

def graficar_matriz_confusion(y_true : np.ndarray,
                              y_pred : np.ndarray,
                              nombres_clases : list[str],
                              titulo : str,
                              ruta_archivo : Path,
                              normalizar : bool = True) :
    mc = confusion_matrix(y_true, y_pred)

    matriz_graficar = mc
    formato = "d"

    if normalizar :
        matriz_graficar = mc.astype(float) / mc.sum(axis = 1, keepdims = True)
        formato = ".2%"

    fig, ax = plt.subplots(figsize = FIGSIZE)

    sns.heatmap(
        data = matriz_graficar,
        annot = True,
        fmt = formato,
        cmap = "Blues",
        xticklabels = nombres_clases,
        yticklabels = nombres_clases,
        cbar = True,
        ax = ax
    )

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    plt.savefig(ruta_archivo, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)