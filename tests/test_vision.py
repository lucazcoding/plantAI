"""Testes do módulo de Visão Computacional (ImageProcessor, MobileNetV2Embedder, VisionAgent) com imagens REAIS."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.vision_agent import VisionAgent
from src.vision.classifier import MobileNetV2Classifier, ModelNotTrainedError
from src.vision.embedder import MULTISCALE_DIM, MobileNetV2Embedder
from src.vision.image_processor import ImageProcessor
from tests.helpers import dark_image, exemplo, flat_image


def test_processor_real_image_shape_range_and_features() -> None:
    res = ImageProcessor().process(exemplo("plantvillage_Tomato_Early_blight.jpg"))
    assert res.is_valid and not res.is_corrupted
    assert res.tensor is not None and res.tensor.shape == (1, 224, 224, 3)
    assert 0.0 <= float(res.tensor.min()) and float(res.tensor.max()) <= 1.0
    assert res.feature_vector is not None and res.feature_vector.shape == (30,)


def test_processor_quality_gates_and_corrupted_input() -> None:
    proc = ImageProcessor()
    assert proc.process(dark_image()).is_too_dark
    assert proc.process(flat_image()).is_blurry
    bad = proc.process(b"nao-e-uma-imagem")
    assert bad.is_corrupted and not bad.is_valid
    assert proc.process(None).is_corrupted


def test_embedder_dimensions() -> None:
    emb = MobileNetV2Embedder()
    img = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    assert emb.embed(img).shape == (1280,)
    assert emb.embed_multiscale(img).shape == (MULTISCALE_DIM,)
    with pytest.raises(ValueError):
        emb.embed(img[:100])


def test_untrained_classifier_refuses_to_invent_probabilities() -> None:
    clf = MobileNetV2Classifier(weights_path=None)
    tensor = ImageProcessor().process(exemplo("plantvillage_Tomato_healthy.jpg")).tensor
    assert tensor is not None
    with pytest.raises(ModelNotTrainedError):
        clf.predict(tensor)


def test_vision_agent_contract_and_correct_class_on_real_examples() -> None:
    agent = VisionAgent()
    expected = {
        "plantvillage_Tomato_Early_blight.jpg": "Tomato___Early_blight",
        "plantvillage_Tomato_Late_blight.jpg": "Tomato___Late_blight",
        "plantvillage_Tomato_healthy.jpg": "Tomato___healthy",
        "plantvillage_Potato_Early_blight.jpg": "Potato___Early_blight",
        "plantvillage_Potato_Late_blight.jpg": "Potato___Late_blight",
        "plantvillage_Corn_maize_Common_rust.jpg": "Corn_(maize)___Common_rust_",
    }
    hits = 0
    for fname, cls in expected.items():
        out = agent.analyze(exemplo(fname))
        assert out["agent_type"] == "vision" and 0.0 <= out["confidence"] <= 1.0 and len(out["top_k"]) == 3
        hits += out["predicted_class"] == cls
    assert hits >= 5, f"apenas {hits}/6 exemplos reais classificados corretamente"
