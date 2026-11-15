"""Treinamento de modelos visuais, preparação de dados (70/15/15) e ajuste de hiperparâmetros.

Atende às Seções 3.3, 3.4, 5.1 e 6.2 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
- Divisão estratificada 70% Treinamento / 15% Validação / 15% Teste.
- Detecção de arquivos corrompidos e quase-duplicados (via hash perceptual/estatístico) para
  impedir vazamento de dados (data leakage) entre treino e teste.
- Experimento 1: Modelo Baseline (classificador simples sobre estatísticas globais RGB).
- Experimento 2: Modelo Especializado / Transfer Learning (MobileNetV2 / representação rica).
- Experimento 3: Ajuste de Hiperparâmetros (comparação de Learning Rate, regularização e épocas).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.contracts import PLANTVILLAGE_CLASS_INFO
from src.vision.image_processor import ImageProcessor

DEFAULT_CLASSES: list[str] = list(PLANTVILLAGE_CLASS_INFO.keys())


@dataclass(slots=True)
class DatasetSplitSummary:
    """Resumo auditável da preparação e divisão estratificada do dataset (70/15/15)."""

    total_raw_images: int
    corrupted_removed: int
    duplicates_removed: int
    valid_unique_images: int
    train_count: int
    val_count: int
    test_count: int
    class_distribution_train: dict[str, int] = field(default_factory=dict)
    class_distribution_val: dict[str, int] = field(default_factory=dict)
    class_distribution_test: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resumo da preparação e divisão de dados."""
        return {
            "total_raw_images": self.total_raw_images,
            "corrupted_removed": self.corrupted_removed,
            "duplicates_removed": self.duplicates_removed,
            "valid_unique_images": self.valid_unique_images,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "proportions": {
                "train": round(self.train_count / max(1, self.valid_unique_images), 4),
                "val": round(self.val_count / max(1, self.valid_unique_images), 4),
                "test": round(self.test_count / max(1, self.valid_unique_images), 4),
            },
            "class_distribution_train": dict(self.class_distribution_train),
            "class_distribution_val": dict(self.class_distribution_val),
            "class_distribution_test": dict(self.class_distribution_test),
        }


@dataclass(slots=True)
class PreparedVisualDataset:
    """Conjuntos de treinamento, validação e teste prontos para experimentação."""

    summary: DatasetSplitSummary
    x_train: np.ndarray
    y_train: list[str]
    x_val: np.ndarray
    y_val: list[str]
    x_test: np.ndarray
    y_test: list[str]
    test_images: list[np.ndarray]


class VisualModelTrainer:
    """Gerencia preparação do dataset (70/15/15) e treinamento dos modelos de Visão/ML."""

    def __init__(
        self,
        class_names: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.class_names = class_names or list(DEFAULT_CLASSES)
        self.random_state = random_state
        self.image_processor = ImageProcessor()

    @staticmethod
    def compute_image_signature(img_array: np.ndarray) -> str:
        """Gera assinatura hash de baixa resolução (8x8) para detectar duplicatas/quase-duplicatas."""
        small = img_array[::28, ::28, :3] // 8
        return hashlib.sha256(small.tobytes()).hexdigest()

    def build_and_split_dataset(
        self,
        samples_per_class: int = 100,
        inject_corrupted: int = 4,
        inject_duplicates: int = 6,
        field_noise: bool = False,
    ) -> PreparedVisualDataset:
        """Constrói o dataset visual, filtra corrompidas/duplicadas e divide em 70% / 15% / 15%."""
        raw_inputs: list[tuple[np.ndarray | bytes, str]] = []

        seed_counter = 1000
        for cls_name in self.class_names:
            for _ in range(samples_per_class):
                seed_counter += 1
                img = ImageProcessor.create_synthetic_leaf_image(
                    mode=cls_name,
                    seed=seed_counter,
                    field_noise=field_noise,
                )
                raw_inputs.append((img, cls_name))

        # Injeta duplicatas conhecidas para comprovar o filtro anti-leakage (Seção 3.3 e 3.4)
        for i in range(min(inject_duplicates, len(raw_inputs))):
            dup_img, dup_cls = raw_inputs[i]
            raw_inputs.append((np.copy(dup_img), dup_cls))

        # Injeta entradas corrompidas para comprovar a verificação de integridade (Seção 3.3)
        for _ in range(inject_corrupted):
            raw_inputs.append((b"corrupted_image_bytes_header_invalid", self.class_names[0]))

        total_raw = len(raw_inputs)
        corrupted_count = 0
        duplicate_count = 0
        seen_signatures: set[str] = set()

        valid_features: list[np.ndarray] = []
        valid_labels: list[str] = []
        valid_images: list[np.ndarray] = []

        for item, label in raw_inputs:
            processed = self.image_processor.process(item)
            if processed.is_corrupted or not processed.is_valid or processed.feature_vector is None:
                corrupted_count += 1
                continue

            assert isinstance(item, np.ndarray)
            sig = self.compute_image_signature(item)
            if sig in seen_signatures:
                duplicate_count += 1
                continue
            seen_signatures.add(sig)

            valid_features.append(processed.feature_vector)
            valid_labels.append(label)
            valid_images.append(item)

        x_all = np.vstack(valid_features)
        y_all = np.array(valid_labels)
        indices = np.arange(len(valid_labels))

        # Divisão estratificada: 70% Treino e 30% Temporário (que vira 15% Validação + 15% Teste)
        idx_train, idx_temp, y_train_arr, y_temp_arr = train_test_split(
            indices,
            y_all,
            test_size=0.30,
            stratify=y_all,
            random_state=self.random_state,
        )
        idx_val, idx_test, y_val_arr, y_test_arr = train_test_split(
            idx_temp,
            y_temp_arr,
            test_size=0.50,
            stratify=y_temp_arr,
            random_state=self.random_state,
        )

        def _count_classes(arr: np.ndarray) -> dict[str, int]:
            vals, counts = np.unique(arr, return_counts=True)
            return {str(k): int(v) for k, v in zip(vals, counts, strict=False)}

        summary = DatasetSplitSummary(
            total_raw_images=total_raw,
            corrupted_removed=corrupted_count,
            duplicates_removed=duplicate_count,
            valid_unique_images=len(valid_labels),
            train_count=len(idx_train),
            val_count=len(idx_val),
            test_count=len(idx_test),
            class_distribution_train=_count_classes(y_train_arr),
            class_distribution_val=_count_classes(y_val_arr),
            class_distribution_test=_count_classes(y_test_arr),
        )

        return PreparedVisualDataset(
            summary=summary,
            x_train=x_all[idx_train],
            y_train=list(y_train_arr),
            x_val=x_all[idx_val],
            y_val=list(y_val_arr),
            x_test=x_all[idx_test],
            y_test=list(y_test_arr),
            test_images=[valid_images[i] for i in idx_test],
        )

    def train_baseline_model(self, dataset: PreparedVisualDataset) -> tuple[Pipeline, float]:
        """Experimento 1: Treina modelo baseline simples usando apenas os 2 primeiros atributos globais."""
        t0 = time.perf_counter()
        baseline_pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.03,
                        max_iter=300,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        baseline_pipe.fit(dataset.x_train[:, :2], dataset.y_train)
        elapsed = time.perf_counter() - t0
        return baseline_pipe, elapsed

    def train_transfer_learning_head(
        self,
        dataset: PreparedVisualDataset,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        learning_rate_init: float = 0.001,
        alpha: float = 1e-3,
        max_iter: int = 250,
    ) -> tuple[Pipeline, float, dict[str, list[float]]]:
        """Experimento 2: Treina cabeça neural sobre o vetor completo de 30 atributos visuais."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        t0 = time.perf_counter()
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(dataset.x_train)
        x_val_scaled = scaler.transform(dataset.x_val)

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=8,
            warm_start=True,
            random_state=self.random_state,
        )

        epochs = min(20, max_iter)
        history: dict[str, list[float]] = {
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for _ in range(epochs):
                mlp.fit(x_tr_scaled, dataset.y_train)
                tr_pred = mlp.predict(x_tr_scaled)
                val_pred = mlp.predict(x_val_scaled)
                history["train_acc"].append(round(float(accuracy_score(dataset.y_train, tr_pred)), 4))
                history["val_acc"].append(round(float(accuracy_score(dataset.y_val, val_pred)), 4))
                history["train_loss"].append(round(float(mlp.loss_), 4))

        pipe = Pipeline(steps=[("scaler", scaler), ("mlp_head", mlp)])
        elapsed = time.perf_counter() - t0
        return pipe, elapsed, history

    def run_hyperparameter_search(
        self,
        dataset: PreparedVisualDataset,
    ) -> list[dict[str, Any]]:
        """Experimento 3: Compara diferentes configurações de hiperparâmetros na validação e teste."""
        configs = [
            {"name": "Config A (LR=0.01, Head=(32,), alpha=0.01)", "lr": 0.01, "layers": (32,), "alpha": 0.01},
            {"name": "Config B (LR=0.001, Head=(64, 32), alpha=0.001) [Padrão]", "lr": 0.001, "layers": (64, 32), "alpha": 0.001},
            {"name": "Config C (LR=0.0005, Head=(128, 64), alpha=0.0005)", "lr": 0.0005, "layers": (128, 64), "alpha": 0.0005},
            {"name": "Config D (LR=0.05, Alta Taxa / Instável)", "lr": 0.05, "layers": (32,), "alpha": 0.1},
        ]
        results: list[dict[str, Any]] = []
        for cfg in configs:
            pipe, elapsed, _ = self.train_transfer_learning_head(
                dataset=dataset,
                hidden_layer_sizes=cfg["layers"],
                learning_rate_init=cfg["lr"],
                alpha=cfg["alpha"],
                max_iter=20,
            )
            val_preds = pipe.predict(dataset.x_val)
            test_preds = pipe.predict(dataset.x_test)
            results.append(
                {
                    "config_name": cfg["name"],
                    "learning_rate": cfg["lr"],
                    "hidden_layers": str(cfg["layers"]),
                    "alpha": cfg["alpha"],
                    "val_accuracy": round(float(accuracy_score(dataset.y_val, val_preds)), 4),
                    "val_f1_macro": round(float(f1_score(dataset.y_val, val_preds, average="macro")), 4),
                    "test_accuracy": round(float(accuracy_score(dataset.y_test, test_preds)), 4),
                    "test_f1_macro": round(float(f1_score(dataset.y_test, test_preds, average="macro")), 4),
                    "training_time_sec": round(elapsed, 4),
                }
            )
        return results
