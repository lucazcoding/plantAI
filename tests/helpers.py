"""Auxiliares de teste. Saídas de visão montadas à mão são FIXTURES para testar as regras de decisão
(unitário); os experimentos científicos e a demo usam apenas o modelo real."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.contracts import AgentOutputDict

ROOT = Path(__file__).resolve().parents[1]
EXEMPLOS = ROOT / "results" / "exemplos"


def exemplo(name: str) -> Path:
    return EXEMPLOS / name


def vision_fixture(probs: dict[str, float]) -> AgentOutputDict:
    """Saída de visão (contrato padronizado) com probabilidades dadas — apenas para testes das regras."""
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "agent_type": "vision",
        "predicted_class": top[0][0],
        "confidence": top[0][1],
        "top_k": top,
        "metadata": {"image_valid": True, "is_corrupted": False, "is_too_dark": False, "is_blurry": False, "quality_issues": []},
    }


def dark_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(2, 12, size=(224, 224, 3), dtype=np.uint8)


def flat_image() -> np.ndarray:
    return np.full((224, 224, 3), 110, dtype=np.uint8)
