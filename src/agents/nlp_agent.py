"""Agente de Processamento de Linguagem Natural (NLPAgent).

Encapsula:
1. `TextProcessor` (limpeza textual, normalização e extração de palavras-chave de sintomas).
2. `SymptomClassifier` (vetorização TF-IDF + LogisticRegression/SVM com contrato padronizado).
"""

from __future__ import annotations

from pathlib import Path

from src.contracts import AgentOutputDict
from src.nlp.symptom_classifier import (
    DEFAULT_DATASET_PATH,
    ClassifierAlgorithm,
    NLPEvaluationMetrics,
    SymptomClassifier,
)
from src.nlp.text_processor import TextProcessor


class NLPAgent:
    """Agente especialista responsável pela interpretação e classificação de sintomas textuais."""

    def __init__(
        self,
        text_processor: TextProcessor | None = None,
        classifier: SymptomClassifier | None = None,
        algorithm: ClassifierAlgorithm = "logistic_regression",
        auto_train_from_sample: bool = True,
        sample_csv_path: str | Path = DEFAULT_DATASET_PATH,
    ) -> None:
        self.text_processor = text_processor or TextProcessor()
        self.classifier = classifier or SymptomClassifier(
            algorithm=algorithm,
            text_processor=self.text_processor,
        )
        if auto_train_from_sample and not self.classifier.is_trained:
            csv_path = Path(sample_csv_path)
            if csv_path.exists():
                self.classifier.train_from_csv(csv_path, evaluate_cv=True)

    @property
    def evaluation_metrics(self) -> NLPEvaluationMetrics | None:
        """Retorna as métricas científicas (Accuracy, Precision, Recall, F1) do modelo treinado."""
        return self.classifier.last_evaluation

    def analyze(
        self,
        symptom_text: str | None,
    ) -> AgentOutputDict:
        """Processa o relato de sintomas do usuário e classifica no formato do contrato padrão."""
        processed = self.text_processor.process(symptom_text)
        return self.classifier.predict(
            text_or_processed=processed,
        )

