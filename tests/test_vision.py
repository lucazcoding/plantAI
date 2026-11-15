"""Testes unitários do módulo de Visão Computacional (`ImageProcessor` e `VisionAgent`)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.agents.vision_agent import VisionAgent
from src.vision.image_processor import ImageProcessor


def test_image_processor_valid_and_feature_extraction() -> None:
    """Verifica redimensionamento 224x224, normalização [0, 1] e extração de 30 atributos visuais."""
    processor = ImageProcessor()
    raw_img = ImageProcessor.create_synthetic_leaf_image("Tomato___Early_blight", size=(300, 260))

    processed = processor.process(raw_img)
    assert processed.is_valid is True
    assert processed.is_corrupted is False
    assert processed.original_size == (300, 260)
    assert processed.tensor is not None
    assert processed.tensor.shape == (1, 224, 224, 3)
    assert 0.0 <= float(np.min(processed.tensor)) <= float(np.max(processed.tensor)) <= 1.0
    assert processed.feature_vector is not None
    assert processed.feature_vector.shape == (30,)
    assert "necrotic_lesion_ratio" in processed.visual_features
    assert len(processed.detected_visual_patterns) >= 1


def test_image_processor_detects_corrupted_dark_blurry_and_low_res() -> None:
    """Garante detecção de bytes corrompidos, imagem escura, imagem borrada e baixa resolução."""
    processor = ImageProcessor()

    corrupted = processor.process(b"invalid_image_bytes_stream")
    assert corrupted.is_corrupted is True
    assert corrupted.is_valid is False
    assert corrupted.tensor is None

    dark = processor.process(ImageProcessor.create_synthetic_leaf_image("dark"))
    assert dark.is_too_dark is True
    assert dark.is_valid is False

    blurry = processor.process(ImageProcessor.create_synthetic_leaf_image("blurry"))
    assert blurry.is_blurry is True
    assert blurry.is_valid is False

    tiny_pil = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))
    tiny_res = processor.process(tiny_pil)
    assert tiny_res.is_valid is False
    assert any("Resolução muito baixa" in msg for msg in tiny_res.quality_issues)


def test_vision_agent_encapsulation_and_contract() -> None:
    """Garante que o VisionAgent processa imagens e devolve o contrato padronizado."""
    agent = VisionAgent()
    rust_leaf = ImageProcessor.create_synthetic_leaf_image("Corn_(maize)___Common_rust_")

    output = agent.analyze(rust_leaf)
    assert output["agent_type"] == "vision"
    assert output["predicted_class"] == "Corn_(maize)___Common_rust_"
    assert 0.0 <= output["confidence"] <= 1.0
    assert len(output["top_k"]) == 3
    assert output["metadata"]["image_valid"] is True
    assert "visual_features" in output["metadata"]

