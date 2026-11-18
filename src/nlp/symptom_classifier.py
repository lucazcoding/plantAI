"""Classificador Supervisionado de Sintomas Textuais (Baseline TF-IDF + LR/SVM).

Atende à Decisão Técnica Inegociável #2:
- O módulo de PLN NÃO é um wrapper de LLM.
- Vetoriza descrições textuais via TF-IDF (unigramas e bigramas) e classifica nas
  categorias do PlantVillage usando LogisticRegression ou SVM calibrado.
- Calcula métricas científicas formais: Accuracy, Precision, Recall, F1-Score e Matriz de Confusão.
- Retorna predições estritamente no contrato padronizado (`AgentOutputDict`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.contracts import AgentOutputDict, validate_agent_output
from src.nlp.text_processor import ProcessedText, TextProcessor

ClassifierAlgorithm = Literal["logistic_regression", "svm"]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "symptoms_sample.csv"


@dataclass(slots=True)
class NLPEvaluationMetrics:
    """Métricas científicas de avaliação do classificador textual."""

    algorithm: str
    num_samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    class_names: list[str]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Converte as métricas de avaliação para dicionário serializável."""
        return {
            "algorithm": self.algorithm,
            "num_samples": self.num_samples,
            "accuracy": round(float(self.accuracy), 4),
            "precision_macro": round(float(self.precision_macro), 4),
            "recall_macro": round(float(self.recall_macro), 4),
            "f1_macro": round(float(self.f1_macro), 4),
            "class_names": list(self.class_names),
            "confusion_matrix": self.confusion_matrix,
        }


class SymptomClassifier:
    """Pipeline de classificação de sintomas textuais com TF-IDF + LogisticRegression/SVM."""

    def __init__(
        self,
        algorithm: ClassifierAlgorithm = "logistic_regression",
        top_k: int = 3,
        random_state: int = 42,
        text_processor: TextProcessor | None = None,
    ) -> None:
        self.algorithm: ClassifierAlgorithm = algorithm
        self.top_k = top_k
        self.random_state = random_state
        self.text_processor = text_processor or TextProcessor()
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []
        self.is_trained: bool = False
        self.last_evaluation: NLPEvaluationMetrics | None = None

    def _build_pipeline(self) -> Pipeline:
        """Constrói o pipeline Scikit-Learn TF-IDF + Classificador Linear."""
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

        if self.algorithm == "svm":
            base_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=self.random_state)
            clf: Any = CalibratedClassifierCV(estimator=base_svm, cv=3)
        else:
            clf = LogisticRegression(
                C=12.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )

        return Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("classifier", clf),
            ]
        )

    def train(self, texts: list[str], labels: list[str]) -> SymptomClassifier:
        """Treina o pipeline TF-IDF + Classificador em textos rotulados."""
        if len(texts) == 0 or len(texts) != len(labels):
            raise ValueError("As listas de textos e rótulos devem ser não vazias e de mesmo tamanho.")

        normalized_texts = [self.text_processor.process(t).normalized_text for t in texts]
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(normalized_texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        self.is_trained = True
        return self

    def train_from_csv(
        self,
        csv_path: str | Path = DEFAULT_DATASET_PATH,
        evaluate_cv: bool = True,
    ) -> NLPEvaluationMetrics | None:
        """Carrega o CSV de sintomas, executa avaliação científica e treina o modelo final."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset de sintomas não encontrado em: {path}")

        df = pd.read_csv(path)
        if "symptom_description" not in df.columns or "disease_class" not in df.columns:
            raise ValueError(
                "O CSV deve conter as colunas 'symptom_description' e 'disease_class'."
            )

        texts = df["symptom_description"].astype(str).tolist()
        labels = df["disease_class"].astype(str).tolist()

        metrics: NLPEvaluationMetrics | None = None
        if evaluate_cv and len(texts) >= 12:
            metrics = self.evaluate_cross_validation(texts, labels, n_splits=5)
            self.last_evaluation = metrics

        self.train(texts, labels)
        return metrics

    def evaluate_split(
        self,
        texts: list[str],
        labels: list[str],
        test_size: float = 0.25,
    ) -> NLPEvaluationMetrics:
        """Avalia o modelo usando divisão estratificada treino/teste."""
        normalized_texts = [self.text_processor.process(t).normalized_text for t in texts]
        x_train, x_test, y_train, y_test = train_test_split(
            normalized_texts,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=self.random_state,
        )
        eval_pipeline = self._build_pipeline()
        eval_pipeline.fit(x_train, y_train)
        y_pred = eval_pipeline.predict(x_test)
        class_names = sorted(set(labels))

        return self._compute_metrics(
            y_true=y_test,
            y_pred=list(y_pred),
            class_names=class_names,
            num_samples=len(y_test),
        )

    def evaluate_cross_validation(
        self,
        texts: list[str],
        labels: list[str],
        n_splits: int = 5,
    ) -> NLPEvaluationMetrics:
        """Avalia o classificador com validação cruzada estratificada (Accuracy, Precision, Recall, F1)."""
        normalized_texts = [self.text_processor.process(t).normalized_text for t in texts]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        eval_pipeline = self._build_pipeline()
        y_pred = cross_val_predict(eval_pipeline, normalized_texts, labels, cv=cv)
        class_names = sorted(set(labels))

        return self._compute_metrics(
            y_true=labels,
            y_pred=list(y_pred),
            class_names=class_names,
            num_samples=len(labels),
        )

    def _compute_metrics(
        self,
        y_true: list[str],
        y_pred: list[str],
        class_names: list[str],
        num_samples: int,
    ) -> NLPEvaluationMetrics:
        """Calcula Accuracy, Precision, Recall, F1-Score e Matriz de Confusão."""
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=class_names).tolist()

        return NLPEvaluationMetrics(
            algorithm=f"TF-IDF + {self.algorithm}",
            num_samples=num_samples,
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1,
            class_names=class_names,
            confusion_matrix=cm,
        )

    def predict(
        self,
        text_or_processed: str | ProcessedText,
    ) -> AgentOutputDict:
        """Classifica a descrição textual de sintomas e retorna no contrato padronizado."""
        processed = (
            text_or_processed
            if isinstance(text_or_processed, ProcessedText)
            else self.text_processor.process(text_or_processed)
        )

        if not self.is_trained or self.pipeline is None:
            if DEFAULT_DATASET_PATH.exists():
                self.train_from_csv(DEFAULT_DATASET_PATH, evaluate_cv=False)
            else:
                raise RuntimeError("SymptomClassifier não está treinado e o CSV padrão não foi encontrado.")

        if processed.is_empty:
            return validate_agent_output(
                {
                    "agent_type": "nlp",
                    "predicted_class": "UNKNOWN",
                    "confidence": 0.0,
                    "top_k": [],
                    "metadata": {
                        **processed.to_metadata(),
                        "model": f"TF-IDF + {self.algorithm}",
                        "warning": "Texto vazio ou sem termos informativos.",
                    },
                }
            )

        proba_vec = self.pipeline.predict_proba([processed.normalized_text])[0]
        probs_by_class = {
            cls_name: float(prob)
            for cls_name, prob in zip(self.classes_, proba_vec, strict=False)
        }

        # Ajuste leve e auditável se o usuário nomeou explicitamente a cultura agrícola
        # (ex: "tomate" vs "batata"), evitando que sintomas idênticos de pinta-preta
        # ou requeima troquem a espécie explicitamente declarada pelo produtor.
        if len(processed.detected_crops) == 1:
            mentioned_crop = processed.detected_crops[0]
            boosted: dict[str, float] = {}
            for cls_name, prob in probs_by_class.items():
                if cls_name.startswith(mentioned_crop):
                    boosted[cls_name] = prob * 1.35
                else:
                    boosted[cls_name] = prob * 0.75
            norm_factor = sum(boosted.values()) or 1.0
            probs_by_class = {k: v / norm_factor for k, v in boosted.items()}

        sorted_probs = sorted(probs_by_class.items(), key=lambda item: item[1], reverse=True)
        top_k_list = [(cls_name, round(float(prob), 4)) for cls_name, prob in sorted_probs[: self.top_k]]
        predicted_class, top_confidence = top_k_list[0]
        backend = f"sklearn_tfidf_{self.algorithm}"

        metadata = {
            **processed.to_metadata(),
            "model": f"TF-IDF + {self.algorithm}",
            "inference_backend": backend,
        }

        output: AgentOutputDict = {
            "agent_type": "nlp",
            "predicted_class": predicted_class,
            "confidence": round(float(top_confidence), 4),
            "top_k": top_k_list,
            "metadata": metadata,
        }
        return validate_agent_output(output)
