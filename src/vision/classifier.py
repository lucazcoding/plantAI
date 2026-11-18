"""Classificador visual: backbone MobileNetV2 pré-treinado + cabeça supervisionada treinada no PlantVillage.

Fluxo real de inferência (sem simulações):
    imagem -> ImageProcessor (224x224, qualidade) -> MobileNetV2 (embedding de 1280-d, ImageNet)
           -> cabeça scikit-learn (StandardScaler + LogisticRegression/MLP) -> probabilidades por classe

Isto é *transfer learning por extração de características* (backbone congelado + cabeça treinada).
O fine-tuning completo do backbone em Keras (GPU) está em `scripts/colab_finetune_mobilenetv2.py` e a
função `build_mobilenetv2_architecture` abaixo é usada por esse script.

Se nenhuma cabeça tiver sido treinada/carregada, `predict` levanta `ModelNotTrainedError`: o sistema
NÃO inventa probabilidades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.contracts import PLANTVILLAGE_CLASS_INFO, AgentOutputDict, validate_agent_output
from src.vision.embedder import MobileNetV2Embedder

DEFAULT_CLASSES: list[str] = list(PLANTVILLAGE_CLASS_INFO.keys())
DEFAULT_HEAD_PATH = Path(__file__).resolve().parents[2] / "models" / "plantai_head.joblib"


class ModelNotTrainedError(RuntimeError):
    """Levantada quando a inferência é pedida sem uma cabeça de classificação treinada."""


class MobileNetV2Classifier:
    """MobileNetV2 (ImageNet, congelado) + cabeça de classificação treinada em imagens reais."""

    def __init__(
        self,
        weights_path: str | Path | None = DEFAULT_HEAD_PATH,
        class_names: list[str] | None = None,
        top_k: int = 3,
        embedder: MobileNetV2Embedder | None = None,
    ) -> None:
        self.class_names = class_names or list(DEFAULT_CLASSES)
        self.top_k = top_k
        self._embedder = embedder
        self.head: Pipeline | None = None
        self.head_description: str = "não treinada"
        self.feature_mode: str = "multiscale"  # "final" (1280-d) ou "multiscale" (1984-d)
        if weights_path is not None and Path(weights_path).exists():
            self.load_head(weights_path)

    @property
    def is_trained(self) -> bool:
        return self.head is not None

    @property
    def embedder(self) -> MobileNetV2Embedder:
        if self._embedder is None:
            self._embedder = MobileNetV2Embedder()
        return self._embedder

    # ------------------------------------------------------------------ treino
    def fit_head(
        self,
        embeddings: np.ndarray,
        labels: list[str] | np.ndarray,
        c: float = 1.0,
        random_state: int = 42,
    ) -> MobileNetV2Classifier:
        """Treina a cabeça (padronização + regressão logística multinomial) sobre embeddings 1280-d."""
        self.head = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=2000, random_state=random_state)),
            ]
        )
        self.head.fit(embeddings, labels)
        self.class_names = list(self.head.classes_)
        self.head_description = f"StandardScaler + LogisticRegression(C={c}) sobre embeddings MobileNetV2"
        return self

    def save_head(self, path: str | Path = DEFAULT_HEAD_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"head": self.head, "description": self.head_description, "feature_mode": self.feature_mode}, path)
        return path

    def load_head(self, path: str | Path = DEFAULT_HEAD_PATH) -> bool:
        payload = joblib.load(path)
        self.head = payload["head"]
        self.head_description = payload.get("description", "cabeça carregada")
        self.feature_mode = payload.get("feature_mode", "multiscale")
        self.class_names = list(self.head.classes_)
        return True

    # --------------------------------------------------------------- inferência
    def predict_proba_embedding(self, embedding: np.ndarray) -> dict[str, float]:
        if self.head is None:
            raise ModelNotTrainedError(
                "Cabeça do classificador não treinada. Rode `python -m src.ml.run_all_experiments` "
                "(gera models/plantai_head.joblib) ou chame fit_head()."
            )
        proba = self.head.predict_proba(embedding.reshape(1, -1))[0]
        return {c: float(p) for c, p in zip(self.class_names, proba, strict=True)}

    def predict(
        self,
        image_tensor: np.ndarray,
        image_metadata: dict[str, Any] | None = None,
    ) -> AgentOutputDict:
        """Inferência real a partir do tensor (1, 224, 224, 3) normalizado em [0, 1]."""
        if image_tensor is None or image_tensor.shape != (1, 224, 224, 3):
            raise ValueError(
                f"image_tensor deve possuir shape (1, 224, 224, 3), recebido: "
                f"{None if image_tensor is None else image_tensor.shape}"
            )
        if self.feature_mode == "multiscale":
            features = self.embedder.embed_multiscale_tensor(image_tensor)
        else:
            features = self.embedder.embed_tensor(image_tensor)
        probs = self.predict_proba_embedding(features)
        return self.build_output(probs, image_metadata)

    def build_output(self, probs_by_class: dict[str, float], image_metadata: dict[str, Any] | None = None) -> AgentOutputDict:
        sorted_probs = sorted(probs_by_class.items(), key=lambda it: it[1], reverse=True)
        top = [(c, round(float(p), 4)) for c, p in sorted_probs[: self.top_k]]
        metadata: dict[str, Any] = {
            "architecture": "MobileNetV2 (ImageNet, congelado) + cabeça treinada",
            "input_resolution": "224x224",
            "inference_backend": "tflite_mobilenetv2_embeddings+sklearn_head",
            "head": self.head_description,
        }
        if image_metadata:
            metadata.update(image_metadata)
        return validate_agent_output(
            {
                "agent_type": "vision",
                "predicted_class": top[0][0],
                "confidence": top[0][1],
                "top_k": top,
                "metadata": metadata,
            }
        )

    # -------------------------------------------------- fine-tuning (Colab / GPU)
    @staticmethod
    def build_mobilenetv2_architecture(
        num_classes: int,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        learning_rate: float = 1e-3,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
    ) -> Any:
        """Modelo Keras MobileNetV2 + cabeça softmax (usado em `scripts/colab_finetune_mobilenetv2.py`)."""
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow não instalado; execute no Google Colab.") from exc

        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = not freeze_backbone
        inputs = tf.keras.Input(shape=input_shape, name="leaf_image")
        x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_rescale")(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_probs")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PlantAI_MobileNetV2")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
