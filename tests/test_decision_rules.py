"""Testes unitários para o contrato de dados e motor de regras/heurísticas (`decision_rules.py`)."""

from __future__ import annotations

import pytest

from src.contracts import (
    AgentOutputDict,
    AgentPrediction,
    DecisionStatus,
    validate_agent_output,
)
from src.decision.decision_rules import DecisionRulesEngine


def _make_vision_output(
    predicted_class: str = "Tomato___Early_blight",
    confidence: float = 0.88,
    top_k: list[tuple[str, float]] | None = None,
    image_valid: bool = True,
    is_corrupted: bool = False,
    is_too_dark: bool = False,
    is_blurry: bool = False,
) -> AgentOutputDict:
    if top_k is None:
        rem = round(max(0.0, 1.0 - confidence), 4)
        alt_cls = (
            "Tomato___Late_blight"
            if predicted_class != "Tomato___Late_blight"
            else "Tomato___Early_blight"
        )
        top_k = [
            (predicted_class, confidence),
            (alt_cls, round(rem * 0.7, 4)),
            ("Tomato___healthy", round(rem * 0.3, 4)),
        ]
    return {
        "agent_type": "vision",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_k,
        "metadata": {
            "image_valid": image_valid,
            "is_corrupted": is_corrupted,
            "is_too_dark": is_too_dark,
            "is_blurry": is_blurry,
            "brightness": 115.0 if not is_too_dark else 8.0,
            "blur_score": 85.0 if not is_blurry else 3.5,
            "quality_issues": [],
        },
    }


def _make_nlp_output(
    predicted_class: str = "Tomato___Early_blight",
    confidence: float = 0.85,
    top_k: list[tuple[str, float]] | None = None,
    extracted_symptoms: list[str] | None = None,
) -> AgentOutputDict:
    if top_k is None:
        rem = round(max(0.0, 1.0 - confidence), 4)
        alt_cls = (
            "Tomato___Late_blight"
            if predicted_class != "Tomato___Late_blight"
            else "Tomato___Early_blight"
        )
        top_k = [
            (predicted_class, confidence),
            (alt_cls, round(rem * 0.7, 4)),
            ("Potato___Early_blight", round(rem * 0.3, 4)),
        ]
    return {
        "agent_type": "nlp",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_k,
        "metadata": {
            "extracted_symptoms": extracted_symptoms or ["aneis_concentricos", "halo_amarelado"],
            "detected_crops": ["Tomato"],
            "is_empty_text": False,
        },
    }


def test_contract_validation_and_dataclass_roundtrip() -> None:
    """Garante que o contrato padronizado valida tipos e converte entre dict e dataclass."""
    valid_dict = _make_vision_output()
    validated = validate_agent_output(valid_dict)
    pred_obj = AgentPrediction.from_dict(validated)
    assert pred_obj.agent_type == "vision"
    assert pred_obj.predicted_class == "Tomato___Early_blight"
    assert pred_obj.to_dict()["predicted_class"] == "Tomato___Early_blight"

    with pytest.raises(ValueError, match="chaves obrigatórias ausentes"):
        validate_agent_output({"agent_type": "vision"})

    with pytest.raises(ValueError, match="fora do intervalo"):
        bad = _make_vision_output(confidence=1.4)
        validate_agent_output(bad)


def test_rule_1_rejects_corrupted_dark_or_blurry_image() -> None:
    """Regra 1: Imagens corrompidas, muito escuras ou borradas devem ser bloqueadas."""
    engine = DecisionRulesEngine()

    dark_vision = _make_vision_output(image_valid=False, is_too_dark=True)
    nlp_out = _make_nlp_output()
    res_dark = engine.evaluate(vision_output=dark_vision, nlp_output=nlp_out)
    assert res_dark.status == DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE
    assert res_dark.final_class is None
    assert res_dark.rule_triggered == "R01_IMAGE_QUALITY_REJECTION"

    blurry_vision = _make_vision_output(image_valid=False, is_blurry=True)
    res_blur = engine.evaluate(vision_output=blurry_vision, nlp_output=None)
    assert res_blur.status == DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE
    assert res_blur.final_class is None


def test_rule_2_vision_only_mode_high_and_low_confidence() -> None:
    """Regra 2: Modo somente visual aplica limiar VISION_CONFIDENCE_THRESHOLD (0.75)."""
    engine = DecisionRulesEngine(vision_confidence_threshold=0.75)

    v_high = _make_vision_output(predicted_class="Corn_(maize)___Common_rust_", confidence=0.86)
    res_high = engine.evaluate(vision_output=v_high, nlp_output=None)
    assert res_high.status == DecisionStatus.DIAGNOSTICO_SOMENTE_VISUAL
    assert res_high.final_class == "Corn_(maize)___Common_rust_"
    assert res_high.operation_mode == "vision_only"

    v_low = _make_vision_output(predicted_class="Corn_(maize)___Common_rust_", confidence=0.62)
    res_low = engine.evaluate(vision_output=v_low, nlp_output=None)
    assert res_low.status == DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA
    assert res_low.final_class is None


def test_rule_3_concordance_boosts_confidence() -> None:
    """Regra 3: Concordância entre Visão e PLN emite CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS e boost."""
    engine = DecisionRulesEngine()
    v_out = _make_vision_output(predicted_class="Tomato___Early_blight", confidence=0.85)
    n_out = _make_nlp_output(predicted_class="Tomato___Early_blight", confidence=0.82)

    decision = engine.evaluate(vision_output=v_out, nlp_output=n_out)
    assert decision.status == DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS
    assert decision.final_class == "Tomato___Early_blight"
    weighted_avg = 0.60 * 0.85 + 0.40 * 0.82
    assert decision.combined_confidence > weighted_avg


def test_rule_4_nlp_disambiguates_moderate_vision() -> None:
    """Regra 4: Visão em dúvida moderada (0.45-0.75) no top_k é desempatada pelo PLN confiante."""
    engine = DecisionRulesEngine()
    v_ambiguous = _make_vision_output(
        predicted_class="Tomato___Early_blight",
        confidence=0.52,
        top_k=[
            ("Tomato___Early_blight", 0.52),
            ("Tomato___Late_blight", 0.43),
            ("Tomato___healthy", 0.05),
        ],
    )
    n_strong_late_blight = _make_nlp_output(
        predicted_class="Tomato___Late_blight",
        confidence=0.89,
        extracted_symptoms=["lesoes_encharcadas", "mofo_esbranquicado"],
    )

    decision = engine.evaluate(vision_output=v_ambiguous, nlp_output=n_strong_late_blight)
    assert decision.status == DecisionStatus.DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL
    assert decision.final_class == "Tomato___Late_blight"
    assert decision.rule_triggered == "R04_NLP_DISAMBIGUATES_MODERATE_VISION"
    assert any("desempatando" in ev.lower() for ev in decision.evidences)


def test_rule_5_divergence_detected_explicitly() -> None:
    """Regra 5: Quando Visão e PLN apontam classes diferentes com alta confiança, expõe conflito."""
    engine = DecisionRulesEngine()
    v_out = _make_vision_output(
        predicted_class="Tomato___healthy",
        confidence=0.91,
        top_k=[
            ("Tomato___healthy", 0.91),
            ("Tomato___Early_blight", 0.06),
            ("Tomato___Late_blight", 0.03),
        ],
    )
    n_out = _make_nlp_output(
        predicted_class="Corn_(maize)___Common_rust_",
        confidence=0.93,
        top_k=[
            ("Corn_(maize)___Common_rust_", 0.93),
            ("Potato___Late_blight", 0.05),
            ("Tomato___Late_blight", 0.02),
        ],
    )

    decision = engine.evaluate(vision_output=v_out, nlp_output=n_out)
    assert decision.status == DecisionStatus.DIVERGENCIA_DETECTADA
    assert decision.final_class is None
    assert len(decision.candidate_hypotheses) == 2
    classes_in_hypotheses = {h[0] for h in decision.candidate_hypotheses}
    assert classes_in_hypotheses == {"Tomato___healthy", "Corn_(maize)___Common_rust_"}


def test_rule_6_general_uncertainty_low_confidence() -> None:
    """Regra 6: Ambos abaixo do limiar mínimo (< 0.45) resulta em INCONCLUSIVO_BAIXA_CONFIANCA."""
    engine = DecisionRulesEngine(min_confidence_threshold=0.45)
    v_low = _make_vision_output(predicted_class="Potato___Early_blight", confidence=0.34)
    n_low = _make_nlp_output(predicted_class="Tomato___Late_blight", confidence=0.31)

    decision = engine.evaluate(vision_output=v_low, nlp_output=n_low)
    assert decision.status == DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA
    assert decision.final_class is None
    assert decision.rule_triggered == "R06_GENERAL_UNCERTAINTY_LOW_CONFIDENCE"

