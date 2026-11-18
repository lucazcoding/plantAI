"""Testes ponta a ponta do Coordenador: visão REAL + PLN REAL + regras. Fixtures de visão só testam as regras."""

from __future__ import annotations

import pytest

from src.agents.coordinator_agent import CoordinatorAgent
from src.contracts import DecisionStatus
from src.nlp.question_intent import QuestionIntentClassifier, split_user_message
from tests.helpers import dark_image, exemplo, vision_fixture


@pytest.fixture(scope="module")
def coordinator() -> CoordinatorAgent:
    return CoordinatorAgent()


def test_nlp_agent_metrics_and_prediction(coordinator: CoordinatorAgent) -> None:
    metrics = coordinator.nlp_agent.evaluation_metrics
    assert metrics is not None and metrics.num_samples >= 150
    assert metrics.accuracy >= 0.75  # CV 5-fold no corpus com textos coloquiais, curtos e sem a cultura citada
    pred = coordinator.nlp_agent.analyze("Folhas de milho cobertas por pequenas pústulas salientes marrom-avermelhadas cor de ferrugem.")
    assert pred["predicted_class"] == "Corn_(maize)___Common_rust_"


def test_real_image_and_symptoms_integrated_concordance(coordinator: CoordinatorAgent) -> None:
    decision = coordinator.diagnose(
        image_input=exemplo("plantvillage_Tomato_Late_blight.jpg"),
        symptom_text="Requeima no tomateiro: manchas encharcadas escuras nas folhas e mofo branco na face inferior.",
    )
    assert decision.status == DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS
    assert decision.final_class == "Tomato___Late_blight"
    assert decision.vision_output is not None and "mock" not in decision.vision_output["metadata"]["inference_backend"]


def test_real_image_with_conflicting_text_is_flagged(coordinator: CoordinatorAgent) -> None:
    decision = coordinator.diagnose(
        image_input=exemplo("plantvillage_Tomato_healthy.jpg"),
        symptom_text="Folhas de milho cobertas por pústulas marrom-avermelhadas cor de ferrugem liberando esporos.",
    )
    assert decision.status == DecisionStatus.DIVERGENCIA_DETECTADA and decision.final_class is None


def test_vision_only_and_quality_rejection(coordinator: CoordinatorAgent) -> None:
    ok = coordinator.diagnose(image_input=exemplo("plantvillage_Potato_Late_blight.jpg"))
    assert ok.operation_mode == "vision_only"
    dark = coordinator.diagnose(image_input=dark_image(), symptom_text="Manchas escuras na folha.")
    assert dark.status == DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE and dark.final_class is None


def test_rule_disambiguation_with_fixture_vision(coordinator: CoordinatorAgent) -> None:
    """Regra 4 (unitário): visão em dúvida moderada (fixture) é desempatada pelo PLN real."""
    decision = coordinator.diagnose(
        vision_override=vision_fixture({"Tomato___Early_blight": 0.51, "Tomato___Late_blight": 0.44, "Tomato___healthy": 0.05}),
        symptom_text="Requeima severa no tomateiro: grandes manchas encharcadas nas bordas e mofo esbranquiçado na face inferior.",
    )
    assert decision.status == DecisionStatus.DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL
    assert decision.final_class == "Tomato___Late_blight"


def test_ablation_reuses_inference_and_returns_three_modes(coordinator: CoordinatorAgent) -> None:
    ab = coordinator.run_ablation_comparison(
        image_input=exemplo("plantvillage_Corn_maize_Common_rust.jpg"),
        symptom_text="Pústulas marrom-avermelhadas salientes nas folhas do milho, soltando pó de ferrugem.",
    )
    assert set(ab) == {"vision_only", "nlp_only", "integrated"}
    assert ab["integrated"].final_class == "Corn_(maize)___Common_rust_"


def test_image_plus_question_flow(coordinator: CoordinatorAgent) -> None:
    resp = coordinator.answer(
        exemplo("plantvillage_Tomato_Early_blight.jpg"),
        "Manchas marrons com anéis concêntricos e halo amarelo nas folhas de baixo. Como tratar?",
    )
    assert resp.intent == "treatment"
    assert resp.decision.final_class == "Tomato___Early_blight"
    assert "Diagnóstico mais provável" in resp.answer
    assert resp.symptom_text and resp.question == "Como tratar?"


def test_question_parsing_and_intent_model() -> None:
    parts = split_user_message("Folha com mancha escura. É grave?")
    assert parts.symptom_text == "Folha com mancha escura." and parts.question == "É grave?"
    clf = QuestionIntentClassifier()
    assert clf.predict("Como posso prevenir essa doença no próximo plantio?")[0] == "prevention"
    assert clf.evaluate_cv()["accuracy"] >= 0.75


def test_history_export(coordinator: CoordinatorAgent, tmp_path) -> None:
    assert coordinator.history
    assert coordinator.export_experimental_history(tmp_path / "h.json").exists()
