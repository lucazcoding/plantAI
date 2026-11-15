"""Avaliação científica, cálculo de métricas, matrizes de confusão e geração de gráficos.

Atende à Seção 6 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
- Métricas obrigatórias: Accuracy, Precision, Recall, F1-score e Matriz de Confusão.
- Registro de configuração, número de amostras, tempo de execução, métricas e erros observados.
- Exportação automática de artefatos para `results/metrics/`, `results/confusion_matrices/` e `results/plots/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(slots=True)
class ModelEvaluationReport:
    """Relatório padronizado de avaliação científica de um experimento."""

    experiment_id: str
    model_name: str
    dataset_name: str
    num_samples: int
    training_time_sec: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    class_names: list[str]
    confusion_matrix: list[list[int]]
    top_errors: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o relatório de avaliação para JSON/dict."""
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "training_time_sec": round(float(self.training_time_sec), 4),
            "accuracy": round(float(self.accuracy), 4),
            "precision_macro": round(float(self.precision_macro), 4),
            "recall_macro": round(float(self.recall_macro), 4),
            "f1_macro": round(float(self.f1_macro), 4),
            "class_names": list(self.class_names),
            "confusion_matrix": self.confusion_matrix,
            "top_errors": list(self.top_errors),
            "extra_metadata": dict(self.extra_metadata),
        }


class ExperimentEvaluator:
    """Avaliador quantitativo e gerador de evidências visuais para os experimentos do PlantAI."""

    def __init__(self, results_root: str | Path | None = None) -> None:
        base = Path(results_root) if results_root else Path(__file__).resolve().parents[2] / "results"
        self.results_root = base
        self.metrics_dir = base / "metrics"
        self.cm_dir = base / "confusion_matrices"
        self.plots_dir = base / "plots"

        for d in (self.metrics_dir, self.cm_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self,
        experiment_id: str,
        model_name: str,
        dataset_name: str,
        y_true: list[str],
        y_pred: list[str],
        class_names: list[str],
        training_time_sec: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ModelEvaluationReport:
        """Calcula todas as métricas científicas obrigatórias e identifica os principais erros."""
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=class_names)

        # Identifica pares de confusão fora da diagonal principal
        error_pairs: list[tuple[int, str, str]] = []
        for i, true_cls in enumerate(class_names):
            for j, pred_cls in enumerate(class_names):
                if i != j and cm[i, j] > 0:
                    error_pairs.append((int(cm[i, j]), true_cls, pred_cls))
        error_pairs.sort(reverse=True)
        top_errors = [
            f"{cnt}x '{t_cls}' classificada como '{p_cls}'"
            for cnt, t_cls, p_cls in error_pairs[:5]
        ]
        if not top_errors:
            top_errors = ["Nenhum erro de classificação observado na amostra avaliada."]

        report = ModelEvaluationReport(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_name=dataset_name,
            num_samples=len(y_true),
            training_time_sec=training_time_sec,
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1,
            class_names=class_names,
            confusion_matrix=cm.tolist(),
            top_errors=top_errors,
            extra_metadata=extra_metadata or {},
        )

        json_path = self.metrics_dir / f"{experiment_id}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    def save_confusion_matrix_plot(
        self,
        report: ModelEvaluationReport,
        filename: str | None = None,
        cmap: str = "Blues",
    ) -> Path:
        """Plota e salva a matriz de confusão como PNG em `results/confusion_matrices/`."""
        short_labels = [c.replace("Corn_(maize)___", "Corn_").replace("___", "\n") for c in report.class_names]
        cm_arr = np.array(report.confusion_matrix)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
        im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j,
                    i,
                    format(int(cm_arr[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(
            f"Matriz de Confusão — {report.model_name}\n"
            f"(Acc={report.accuracy:.2%} | F1={report.f1_macro:.2%} | N={report.num_samples})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Classe Prevista", fontsize=10)
        ax.set_ylabel("Classe Verdadeira", fontsize=10)
        plt.tight_layout()

        out_path = self.cm_dir / (filename or f"{report.experiment_id}_cm.png")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
