"""Testes ponta a ponta do Agente Coordenador, Agente Visual, Agente PLN e Relatórios."""

from __future__ import annotations

import pytest

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.vision_agent import VisionAgent
from src.contracts import DecisionStatus
from src.vision.image_processor import ImageProcessor


@pytest.fixture(scope="module")
def coordinator() -> CoordinatorAgent:
    """Instancia o CoordinatorAgent com o NLPAgent treinado no symptoms_sample.csv real."""
    nlp_agent = NLPAgent(auto_train_from_sample=True)
    vision_agent = VisionAgent()
    return CoordinatorAgent(vision_agent=vision_agent, nlp_agent=nlp_agent)


def test_nlp_agent_real_training_and_metrics(coordinator: CoordinatorAgent) -> None:
    """Verifica se o classificador de PLN treinou no CSV e atingiu métricas científicas altas."""
    metrics = coordinator.nlp_agent.evaluation_metrics
    assert metrics is not None
    assert metrics.num_samples >= 40
    assert metrics.accuracy >= 0.85
    assert metrics.f1_macro >= 0.85

    pred = coordinator.nlp_agent.analyze(
        "Folhas de milho cobertas por pequenas pústulas salientes marrom-avermelhadas cor de ferrugem."
    )
    assert pred["agent_type"] == "nlp"
    assert pred["predicted_class"] == "Corn_(maize)___Common_rust_"
    assert pred["confidence"] > 0.60
    assert "pustulas_ferruginosas" in pred["metadata"]["extracted_symptoms"]


def test_coordinator_end_to_end_concordance(coordinator: CoordinatorAgent) -> None:
    """Cenário 1: Concordância total entre imagem e texto real de Pinta-Preta no tomate."""
    leaf_img = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")
    decision = coordinator.diagnose(
        image_input=leaf_img,
        symptom_text=(
            "Folhas velhas do tomateiro com manchas escuras arredondadas formando anéis "
            "concêntricos tipo alvo e halo amarelado ao redor."
        ),
        mock_vision_probs={
            "Tomato___Early_blight": 0.89,
            "Tomato___Late_blight": 0.07,
            "Tomato___Healthy": 0.04,
        },
    )

    assert decision.status == DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS
    assert decision.final_class == "Tomato___Early_blight"
    report_text = coordinator.generate_report(decision, title="Cenário 1")
    assert "CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS" in report_text
    assert "Pinta-preta" in report_text


def test_coordinator_end_to_end_nlp_disambiguation(coordinator: CoordinatorAgent) -> None:
    """Cenário 2: Visão em dúvida moderada entre Early_blight e Late_blight, desempatada pelo PLN."""
    leaf_img = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")
    decision = coordinator.diagnose(
        image_input=leaf_img,
        symptom_text=(
            "Requeima severa no tomateiro: grandes manchas encharcadas de aspecto oleoso nas "
            "bordas das folhas e mofo esbranquiçado na face inferior."
        ),
        mock_vision_probs={
            "Tomato___Early_blight": 0.51,
            "Tomato___Late_blight": 0.44,
            "Tomato___Healthy": 0.05,
        },
    )

    assert decision.status == DecisionStatus.DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL
    assert decision.final_class == "Tomato___Late_blight"


def test_coordinator_end_to_end_divergence(coordinator: CoordinatorAgent) -> None:
    """Cenário 3: Divergência crítica entre imagem (Tomate Saudável) e texto (Ferrugem do Milho)."""
    leaf_img = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")
    decision = coordinator.diagnose(
        image_input=leaf_img,
        symptom_text=(
            "Folhas de milho cobertas por pústulas salientes marrom-avermelhadas cor de canela "
            "liberando esporos de ferrugem nas duas faces."
        ),
        mock_vision_probs={
            "Tomato___Healthy": 0.92,
            "Tomato___Early_blight": 0.05,
            "Tomato___Late_blight": 0.03,
        },
    )

    assert decision.status == DecisionStatus.DIVERGENCIA_DETECTADA
    assert decision.final_class is None
    assert len(decision.candidate_hypotheses) == 2


def test_coordinator_end_to_end_vision_only_and_dark_image(coordinator: CoordinatorAgent) -> None:
    """Cenário 4: Operação somente com imagem (sem texto) e bloqueio de imagem escura."""
    leaf_img = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")
    decision_vis = coordinator.diagnose(
        image_input=leaf_img,
        symptom_text="",
        mock_vision_probs={
            "Potato___Late_blight": 0.84,
            "Potato___Early_blight": 0.12,
            "Tomato___Late_blight": 0.04,
        },
    )
    assert decision_vis.status == DecisionStatus.DIAGNOSTICO_SOMENTE_VISUAL
    assert decision_vis.final_class == "Potato___Late_blight"

    dark_img = ImageProcessor.create_synthetic_leaf_image("dark")
    decision_dark = coordinator.diagnose(
        image_input=dark_img,
        symptom_text="Manchas escuras na folha.",
    )
    assert decision_dark.status == DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE
    assert decision_dark.final_class is None


def test_coordinator_ablation_study_comparison(coordinator: CoordinatorAgent) -> None:
    """Comprova o estudo de ablação comparando Visão Isolada, PLN Isolado e Sistema Integrado."""
    leaf_img = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")
    ablation = coordinator.run_ablation_comparison(
        image_input=leaf_img,
        symptom_text=(
            "Folhas de batata com manchas encharcadas escuras nas margens e mofo branco "
            "na face inferior, típicas de requeima da batateira."
        ),
        mock_vision_probs={
            "Potato___Early_blight": 0.53,
            "Potato___Late_blight": 0.42,
            "Tomato___Late_blight": 0.05,
        },
    )

    assert set(ablation.keys()) == {"vision_only", "nlp_only", "integrated"}
    # Na visão isolada (0.53 < 0.75), o sistema sozinho fica inconclusivo por baixa confiança:
    assert ablation["vision_only"].status == DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA
    # No PLN isolado, acerta Potato___Late_blight:
    assert ablation["nlp_only"].final_class == "Potato___Late_blight"
    # No sistema integrado, desempata a dúvida visual e conclui Potato___Late_blight:
    assert ablation["integrated"].status == DecisionStatus.DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL
    assert ablation["integrated"].final_class == "Potato___Late_blight"

