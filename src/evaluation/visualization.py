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

def graficar_curva_roc_multiclase(y_true: np.ndarray,
                                  y_proba: np.ndarray,
                                  nombres_clases: list[str],
                                  titulo: str,
                                  ruta_archivo: Path,
                                 ) -> None:

    clases = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=clases)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, nombre in enumerate(nombres_clases):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_clase = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_clase:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio (AUC = 0.500)")

    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(ruta_archivo, dpi=300)
    plt.close(fig)

def graficar_curvas_aprendizaje(
                                history: dict,
                                ruta_archivo: Path,
                                titulo: str = "Curvas de aprendizaje (MLP)",
                                ) :
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validación")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validación")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(ruta_archivo, dpi=300)
    plt.close(fig)

def graficar_roc_comparativa(
                             modelos_y_probas: dict[str, tuple[np.ndarray, np.ndarray]],
                             nombres_clases: list[str],
                             ruta_archivo: Path,
                            ) :
    n_clases = len(nombres_clases)

    # 1. Crear la figura base
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for nombre, (y_true, y_proba) in modelos_y_probas.items():

        clases_unicas = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=clases_unicas)

        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_clases):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_clases
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"{nombre} (AUC = {macro_auc:.3f})",
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio (AUC = 0.500)")

    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
    ax.set_title(
        "Curva ROC macro-promedio: comparativa entre modelos",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    plt.savefig(ruta_archivo, dpi=DPI, bbox_inches="tight")
    plt.close(fig)