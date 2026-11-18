"""Testes unitários do módulo de Processamento de Linguagem Natural (`TextProcessor` e `SymptomClassifier`)."""

from __future__ import annotations

from src.agents.nlp_agent import NLPAgent
from src.nlp.symptom_classifier import SymptomClassifier
from src.nlp.text_processor import TextProcessor


def test_text_processor_cleaning_negation_and_extraction() -> None:
    """Testa normalização, preservação de negações ('sem manchas') e extração de sintomas."""
    tp = TextProcessor()

    healthy = tp.process(
        "Folha de tomateiro totalmente saudável, verde uniforme, sem manchas escuras e sem lesões."
    )
    assert healthy.is_empty is False
    assert "Tomato" in healthy.detected_crops
    assert "folha_saudavel" in healthy.extracted_symptoms
    # Não deve extrair 'necrose_foliar' quando há negação explícita ('sem manchas escuras')
    assert "necrose_foliar" not in healthy.extracted_symptoms

    diseased = tp.process(
        "Batateira com lesões circulares apresentando anéis concêntricos tipo alvo e halo amarelado."
    )
    assert "Potato" in diseased.detected_crops
    assert "aneis_concentricos" in diseased.extracted_symptoms
    assert "halo_amarelado" in diseased.extracted_symptoms

    empty = tp.process("   ")
    assert empty.is_empty is True
    assert empty.tokens == []


def test_symptom_classifier_logistic_regression_and_svm() -> None:
    """Avalia o treinamento e predição com LogisticRegression e SVM Calibrado."""
    clf_lr = SymptomClassifier(algorithm="logistic_regression")
    metrics_lr = clf_lr.train_from_csv(evaluate_cv=True)
    assert metrics_lr is not None
    assert metrics_lr.accuracy >= 0.75
    assert metrics_lr.f1_macro >= 0.75

    clf_svm = SymptomClassifier(algorithm="svm")
    metrics_svm = clf_svm.train_from_csv(evaluate_cv=True)
    assert metrics_svm is not None
    assert metrics_svm.accuracy >= 0.75

    pred = clf_svm.predict(
        "Folhas de batata com grandes manchas encharcadas escuras e mofo branco na página inferior."
    )
    assert pred["agent_type"] == "nlp"
    assert pred["predicted_class"] == "Potato___Late_blight"
    assert pred["confidence"] > 0.55


def test_nlp_agent_handles_empty_input_gracefully() -> None:
    """Garante retorno seguro no contrato quando o usuário não digita descrição textual."""
    agent = NLPAgent()
    res = agent.analyze("")
    assert res["agent_type"] == "nlp"
    assert res["predicted_class"] == "UNKNOWN"
    assert res["confidence"] == 0.0
    assert res["metadata"]["is_empty_text"] is True



def test_symptom_classifier_crop_boost_flag_changes_only_culture_mentions() -> None:
    texts = [
        "manchas escuras com anéis concêntricos nas folhas do tomate",
        "manchas encharcadas que escurecem rápido nas folhas da batata",
        "folha verde sem manchas no tomate",
        "pústulas alaranjadas nas folhas do milho",
    ] * 6
    labels = ["Tomato___Early_blight", "Potato___Late_blight", "Tomato___healthy", "Corn_(maize)___Common_rust_"] * 6
    boosted = SymptomClassifier(crop_boost=True).train(texts, labels)
    pure = SymptomClassifier(crop_boost=False).train(texts, labels)
    q = "manchas escuras na folha do tomate"
    pb, pp = boosted.predict(q), pure.predict(q)
    assert pb["predicted_class"].startswith("Tomato")
    # sem reforço a confiança no top-1 não pode ser maior que com reforço na cultura citada
    assert pb["confidence"] >= pp["confidence"] or pb["predicted_class"] != pp["predicted_class"]
    no_crop = "manchas escuras com anéis"
    assert boosted.predict(no_crop)["top_k"] == pure.predict(no_crop)["top_k"]
