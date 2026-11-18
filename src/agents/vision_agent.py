"""Agente 1 — Visão Computacional (`VisionAgent`).

Atende à Seção 4.2 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
1. Receber a imagem.
2. Verificar o formato e integridade.
3. Redimensionar a imagem (224x224).
4. Normalizar os pixels ([0.0, 1.0]).
5. Preparar os dados para o modelo.
6. Extrair características visuais.
7. Identificar padrões relevantes.
8. Encaminhar os resultados ao classificador (`MLAgent` / `MobileNetV2Classifier`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.agents.ml_agent import MLAgent
from src.contracts import AgentOutputDict
from src.vision.classifier import DEFAULT_HEAD_PATH, MobileNetV2Classifier
from src.vision.image_processor import ImageProcessor, ProcessedImage


class VisionAgent:
    """Agente especialista responsável pelo processamento visual e encaminhamento ao MLAgent."""

    def __init__(
        self,
        image_processor: ImageProcessor | None = None,
        classifier: MobileNetV2Classifier | None = None,
        ml_agent: MLAgent | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.image_processor = image_processor or ImageProcessor()
        self.classifier = classifier or MobileNetV2Classifier(weights_path=weights_path or DEFAULT_HEAD_PATH)
        self.ml_agent = ml_agent or MLAgent(classifier=self.classifier)

    def extract_and_prepare(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None,
    ) -> ProcessedImage:
        """Executa as etapas 1 a 7 de Visão Computacional (inspeção, normalização e extração)."""
        return self.image_processor.process(image_input)

    def analyze(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None,
    ) -> AgentOutputDict:
        """Processa a imagem e encaminha ao MLAgent (etapa 8) para obter o contrato padrão."""
        processed = self.extract_and_prepare(image_input)
        return self.ml_agent.classify_processed_image(processed_image=processed)
