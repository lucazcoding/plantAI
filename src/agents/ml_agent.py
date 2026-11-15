"""Agente 2 — Machine Learning (`MLAgent`).

Atende à Seção 4.3 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
- Responsabilidade: classificar as imagens processadas pelo Agente de Visão Computacional
  e identificar a categoria mais provável utilizando MobileNetV2 (Transfer Learning) ou
  classificador treinado sobre as representações visuais.
- Retorna: classe prevista, probabilidades por classe (`top_k`) e confiança associada,
  seguindo estritamente o contrato padronizado (`AgentOutputDict`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import AgentOutputDict, validate_agent_output
from src.vision.classifier import MobileNetV2Classifier
from src.vision.image_processor import ProcessedImage


class MLAgent:
    """Agente especialista de Machine Learning para classificação supervisionada foliar."""

    def __init__(
        self,
        classifier: MobileNetV2Classifier | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.classifier = classifier or MobileNetV2Classifier(weights_path=weights_path)

    def classify_processed_image(
        self,
        processed_image: ProcessedImage,
        mock_probabilities: dict[str, float] | None = None,
    ) -> AgentOutputDict:
        """Recebe a saída do Agente de Visão (`ProcessedImage`) e executa a classificação ML."""
        img_meta = processed_image.to_metadata()

        if processed_image.is_corrupted or processed_image.tensor is None:
            output: AgentOutputDict = {
                "agent_type": "vision",
                "predicted_class": "UNREADABLE_IMAGE",
                "confidence": 0.0,
                "top_k": [],
                "metadata": {
                    **img_meta,
                    "ml_agent": "MLAgent",
                    "architecture": "MobileNetV2",
                    "inference_skipped": True,
                },
            }
            return validate_agent_output(output)

        prediction = self.classifier.predict(
            image_tensor=processed_image.tensor,
            image_metadata=img_meta,
            mock_probabilities=mock_probabilities,
        )
        prediction["metadata"]["ml_agent"] = "MLAgent"
        return prediction

    def fit_from_features(
        self,
        feature_matrix: np.ndarray,
        labels: list[str],
        **kwargs: Any,
    ) -> MLAgent:
        """Treina o classificador interno do MLAgent sobre matriz de atributos visuais."""
        self.classifier.fit_feature_representation(feature_matrix, labels, **kwargs)
        return self

