"""Interpretação da PERGUNTA do usuário (reconhecimento de intenção) — Seção 4.4/4.5 do PDF.

Fluxo exigido pelo enunciado: "o usuário envia uma imagem; o sistema extrai informações visuais; o usuário
faz uma pergunta; o PLN interpreta a solicitação; um componente de decisão combina os resultados".

1. `split_user_message` separa a mensagem em (a) relato de sintomas e (b) pergunta, por regras sobre a
   pontuação e palavras interrogativas.
2. `QuestionIntentClassifier` (TF-IDF + LogisticRegression) reconhece a intenção da pergunta:
   diagnose | treatment | prevention | severity | confirm.
3. O coordenador usa a intenção para escolher qual parte do conhecimento fitossanitário responder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.nlp.text_processor import TextProcessor

Intent = Literal["diagnose", "treatment", "prevention", "severity", "confirm"]
DEFAULT_INTENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "question_intents.csv"

_INTERROGATIVES = (
    "como", "qual", "quais", "o que", "que ", "por que", "porque", "quando", "onde", "isso", "essa", "será",
    "posso", "preciso", "devo", "tem como", "é ", "pode", "vai", "corre", "voce", "você",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_INLINE_QUESTION_START = re.compile(
    r"\b(?:isso (?:é|pode ser)|é grave|como (?:devo |posso )?(?:tratar|evitar|prevenir)|"
    r"qual doença|que doença|será que|posso|devo)\b",
    re.I,
)


@dataclass(slots=True)
class UserMessage:
    """Mensagem do usuário decomposta em relato de sintomas e pergunta."""

    symptom_text: str
    question: str


def split_user_message(text: str | None) -> UserMessage:
    """Separa frases-pergunta (terminam em '?' ou iniciam com interrogativa) do relato de sintomas."""
    if not text or not text.strip():
        return UserMessage("", "")
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    symptoms: list[str] = []
    questions: list[str] = []
    for s in sentences:
        low = s.lower()
        if s.endswith("?") and "," in s and not low.startswith(_INTERROGATIVES):
            symptom_clause, question_clause = s.rsplit(",", 1)
            if question_clause.strip().lower().startswith(_INTERROGATIVES):
                symptoms.append(symptom_clause.strip())
                questions.append(question_clause.strip())
                continue
        if s.endswith("?") and not low.startswith(_INTERROGATIVES):
            question_start = _INLINE_QUESTION_START.search(s)
            if question_start:
                symptoms.append(s[:question_start.start()].strip(" ,;"))
                questions.append(s[question_start.start():].strip())
                continue
        if s.endswith("?") or low.startswith(_INTERROGATIVES):
            questions.append(s)
        else:
            symptoms.append(s)
    return UserMessage(" ".join(symptoms), " ".join(questions))


class QuestionIntentClassifier:
    """Classificador de intenção da pergunta (TF-IDF + regressão logística)."""

    def __init__(self, random_state: int = 42) -> None:
        self.text_processor = TextProcessor()
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=20.0, max_iter=2000, random_state=random_state)),
            ]
        )
        self.is_trained = False

    def _prep(self, texts: list[str]) -> list[str]:
        return [self.text_processor.strip_accents(t.lower()) for t in texts]

    def train_from_csv(self, path: str | Path = DEFAULT_INTENTS_PATH) -> QuestionIntentClassifier:
        df = pd.read_csv(path)
        self.pipeline.fit(self._prep(df["question"].tolist()), df["intent"].tolist())
        self.is_trained = True
        return self

    def evaluate_cv(self, path: str | Path = DEFAULT_INTENTS_PATH, n_splits: int = 5) -> dict[str, float]:
        """Validação cruzada estratificada (as perguntas avaliadas nunca estão no treino do fold)."""
        df = pd.read_csv(path)
        x, y = self._prep(df["question"].tolist()), df["intent"].tolist()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        pred = cross_val_predict(self.pipeline, x, y, cv=skf)
        return {
            "n": len(y),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        }

    def predict(self, question: str) -> tuple[Intent, float]:
        if not self.is_trained:
            self.train_from_csv()
        x = self._prep([question])
        proba = self.pipeline.predict_proba(x)[0]
        idx = int(proba.argmax())
        return str(self.pipeline.classes_[idx]), float(proba[idx])  # type: ignore[return-value]
