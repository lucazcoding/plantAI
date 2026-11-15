"""Classificador Visual baseado em MobileNetV2 (Transfer Learning) e Representação de Atributos.

Projetado para:
1. Construir e carregar a arquitetura MobileNetV2 (`224x224x3`) no Google Colab (TensorFlow/Keras).
2. Permitir treinamento supervisionado local sobre os descritores/embeddings visuais extraídos
   pelo `ImageProcessor` (Scikit-Learn MLP / Calibrated Linear Classifier), garantindo execução
   real de todos os experimentos científicos mesmo em ambientes sem GPU/TensorFlow.
3. Operar em modo `mock/fallback` determinístico caso nenhum modelo tenha sido treinado ou
   quando probabilidades de simulação (`mock_probabilities`) forem injetadas em testes.
4. Retornar sempre no formato estrito do contrato padronizado (`AgentOutputDict`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.contracts import PLANTVILLAGE_CLASS_INFO, AgentOutputDict, validate_agent_output
from src.vision.image_processor import ImageProcessor

DEFAULT_CLASSES: list[str] = list(PLANTVILLAGE_CLASS_INFO.keys())


class MobileNetV2Classifier:
    """Encapsula o classificador visual (MobileNetV2 Keras ou MLP sobre descritores visuais)."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        class_names: list[str] | None = None,
        top_k: int = 3,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.class_names = class_names or list(DEFAULT_CLASSES)
        self.top_k = top_k
        self.model: Any | None = None
        self.feature_classifier: Pipeline | None = None
        self.is_mock_mode: bool = True

        if self.weights_path and self.weights_path.exists():
            self.load_weights(self.weights_path)

    @staticmethod
    def build_mobilenetv2_architecture(
        num_classes: int,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        learning_rate: float = 1e-3,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
    ) -> Any:
        """Constrói o modelo MobileNetV2 com Transfer Learning (ImageNet) em Keras.

        Segue a Seção 5.1 da documentação:
        - Imagem 224x224, Otimizador Adam, learning_rate inicial 0.001, Categorical Crossentropy.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow não está instalado neste ambiente. Execute no Google Colab "
                "ou utilize fit_feature_representation() para treinar localmente."
            ) from exc

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
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

    def fit_feature_representation(
        self,
        feature_matrix: np.ndarray,
        labels: list[str],
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> MobileNetV2Classifier:
        """Treina a cabeça de classificação neural sobre as representações visuais extraídas."""
        self.feature_classifier = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp_head",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=learning_rate_init,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.feature_classifier.fit(feature_matrix, labels)
        self.class_names = list(self.feature_classifier.classes_)
        self.is_mock_mode = False
        return self

    def load_weights(self, weights_path: str | Path) -> bool:
        """Carrega modelo/pesos salvos (.keras, .h5 ou .joblib) caso disponíveis."""
        path = Path(weights_path)
        if not path.exists():
            self.is_mock_mode = True
            return False

        if path.suffix in (".joblib", ".pkl"):
            try:
                import joblib

                loaded = joblib.load(path)
                self.feature_classifier = loaded
                self.class_names = list(loaded.classes_)
                self.is_mock_mode = False
                return True
            except Exception:
                self.is_mock_mode = True
                return False

        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(str(path))
            self.is_mock_mode = False
            return True
        except Exception:
            self.model = None
            self.is_mock_mode = True
            return False

    def predict(
        self,
        image_tensor: np.ndarray,
        image_metadata: dict[str, Any] | None = None,
        mock_probabilities: dict[str, float] | None = None,
    ) -> AgentOutputDict:
        """Realiza a inferência na imagem pré-processada e retorna no contrato padronizado."""
        if image_tensor is None or image_tensor.shape != (1, 224, 224, 3):
            raise ValueError(
                f"image_tensor deve possuir shape (1, 224, 224, 3), recebido: "
                f"{None if image_tensor is None else image_tensor.shape}"
            )

        probs_by_class: dict[str, float]

        if mock_probabilities is not None:
            total = sum(mock_probabilities.values())
            if total <= 0:
                raise ValueError("As probabilidades simuladas devem ter soma positiva.")
            probs_by_class = {k: float(v) / total for k, v in mock_probabilities.items()}
            for cls_name in self.class_names:
                probs_by_class.setdefault(cls_name, 0.0)
            inference_backend = "mock_scenario_override"
        elif not self.is_mock_mode and self.model is not None:
            raw_preds = self.model.predict(image_tensor, verbose=0)[0]
            probs_by_class = {
                cls_name: float(prob)
                for cls_name, prob in zip(self.class_names, raw_preds, strict=False)
            }
            inference_backend = "tensorflow_mobilenetv2"
        elif not self.is_mock_mode and self.feature_classifier is not None:
            _, feat_vec, _ = ImageProcessor.extract_visual_features(image_tensor[0])
            proba = self.feature_classifier.predict_proba(feat_vec.reshape(1, -1))[0]
            probs_by_class = {
                cls_name: float(prob)
                for cls_name, prob in zip(self.class_names, proba, strict=False)
            }
            inference_backend = "trained_visual_mlp_head"
        else:
            probs_by_class = self._deterministic_fallback_probs(image_tensor)
            inference_backend = "mobilenetv2_fallback_simulator"

        sorted_probs = sorted(probs_by_class.items(), key=lambda item: item[1], reverse=True)
        top_k_list = [(cls_name, round(float(prob), 4)) for cls_name, prob in sorted_probs[: self.top_k]]
        predicted_class, top_confidence = top_k_list[0]

        metadata: dict[str, Any] = {
            "architecture": "MobileNetV2",
            "input_resolution": "224x224",
            "inference_backend": inference_backend,
            "is_mock_mode": self.is_mock_mode or (mock_probabilities is not None),
        }
        if image_metadata:
            metadata.update(image_metadata)

        output: AgentOutputDict = {
            "agent_type": "vision",
            "predicted_class": predicted_class,
            "confidence": round(float(top_confidence), 4),
            "top_k": top_k_list,
            "metadata": metadata,
        }
        return validate_agent_output(output)

    def _deterministic_fallback_probs(self, image_tensor: np.ndarray) -> dict[str, float]:
        """Gera distribuição de probabilidades determinística baseada nos descritores visuais."""
        feats, _, _ = ImageProcessor.extract_visual_features(image_tensor[0])
        if feats["rust_pustule_ratio"] > 0.02:
            return {
                "Corn_(maize)___Common_rust_": 0.86,
                "Tomato___Early_blight": 0.06,
                "Potato___Early_blight": 0.04,
                "Tomato___Late_blight": 0.02,
                "Potato___Late_blight": 0.01,
                "Tomato___Healthy": 0.01,
            }
        if feats["water_soaked_ratio"] > 0.02:
            return {
                "Tomato___Late_blight": 0.83,
                "Potato___Late_blight": 0.09,
                "Tomato___Early_blight": 0.04,
                "Potato___Early_blight": 0.02,
                "Tomato___Healthy": 0.01,
                "Corn_(maize)___Common_rust_": 0.01,
            }
        if feats["necrotic_lesion_ratio"] > 0.02:
            return {
                "Tomato___Early_blight": 0.84,
                "Potato___Early_blight": 0.08,
                "Tomato___Late_blight": 0.04,
                "Potato___Late_blight": 0.02,
                "Tomato___Healthy": 0.01,
                "Corn_(maize)___Common_rust_": 0.01,
            }
        return {
            "Tomato___Healthy": 0.85,
            "Tomato___Early_blight": 0.07,
            "Tomato___Late_blight": 0.04,
            "Potato___Early_blight": 0.02,
            "Potato___Late_blight": 0.01,
            "Corn_(maize)___Common_rust_": 0.01,
        }
