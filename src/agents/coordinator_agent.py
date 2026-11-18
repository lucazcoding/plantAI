"""Agente 4 — Coordenador Central do Sistema Multiagente PlantAI (`CoordinatorAgent`).

Fluxo funcional integrado (Seção 4.5 do PDF e Funcionalidade F08):
1. O usuário envia uma imagem e uma mensagem (sintomas e/ou pergunta).
2. `split_user_message` separa relato de sintomas e pergunta.
3. VisionAgent -> MLAgent extraem/classificam a imagem (modelo REAL treinado).
4. NLPAgent classifica os sintomas; `QuestionIntentClassifier` interpreta a pergunta.
5. `DecisionRulesEngine` (6 regras explícitas) combina visão + texto.
6. O coordenador responde à pergunta (diagnóstico/tratamento/prevenção/gravidade/confirmação) com base
   na decisão consolidada e registra o histórico experimental.

Não há probabilidades simuladas aqui. Para testes unitários das regras, use `vision_override` /
`nlp_override` com saídas no contrato padronizado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.agents.ml_agent import MLAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.vision_agent import VisionAgent
from src.contracts import (
    CLASS_QA_INFO,
    AgentOutputDict,
    DecisionResult,
    DecisionStatus,
    OperationMode,
    get_disease_metadata,
    validate_agent_output,
)
from src.decision.decision_rules import DecisionRulesEngine
from src.nlp.question_intent import QuestionIntentClassifier, split_user_message
from src.reports.report_generator import ReportGenerator


@dataclass(slots=True)
class QAResponse:
    """Resposta contextualizada a uma pergunta do usuário sobre a imagem enviada."""

    answer: str
    intent: str
    intent_confidence: float
    question: str
    symptom_text: str
    decision: DecisionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "question": self.question,
            "symptom_text": self.symptom_text,
            "decision": self.decision.to_dict(),
        }


class CoordinatorAgent:
    """Agente Coordenador responsável pela fusão de evidências, resolução de conflitos e ablação."""

    def __init__(
        self,
        vision_agent: VisionAgent | None = None,
        ml_agent: MLAgent | None = None,
        nlp_agent: NLPAgent | None = None,
        decision_engine: DecisionRulesEngine | None = None,
        intent_classifier: QuestionIntentClassifier | None = None,
    ) -> None:
        self.vision_agent = vision_agent or VisionAgent(ml_agent=ml_agent)
        self.ml_agent = ml_agent or self.vision_agent.ml_agent
        self.nlp_agent = nlp_agent or NLPAgent()
        self.decision_engine = decision_engine or DecisionRulesEngine()
        self.intent_classifier = intent_classifier or QuestionIntentClassifier()
        self.history: list[DecisionResult] = []

    # ------------------------------------------------------------------ decisão
    def diagnose(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None = None,
        symptom_text: str | None = None,
        mode: OperationMode | None = None,
        vision_override: AgentOutputDict | None = None,
        nlp_override: AgentOutputDict | None = None,
    ) -> DecisionResult:
        """Executa visão + PLN + regras e retorna a decisão consolidada.

        `vision_override`/`nlp_override` aceitam saídas já calculadas (útil para reutilizar a inferência
        em estudos de ablação e em testes das regras); nunca são gerados por simulação neste módulo.
        """
        v_output: AgentOutputDict | None = None
        n_output: AgentOutputDict | None = None

        run_vision = mode in (None, "integrated", "vision_only")
        run_nlp = mode in (None, "integrated", "nlp_only")

        if run_vision:
            if vision_override is not None:
                v_output = validate_agent_output(vision_override)
            elif image_input is not None:
                processed_img = self.vision_agent.extract_and_prepare(image_input)
                v_output = self.ml_agent.classify_processed_image(processed_image=processed_img)

        if run_nlp:
            if nlp_override is not None:
                n_output = validate_agent_output(nlp_override)
            elif symptom_text is not None and str(symptom_text).strip() != "":
                n_output = self.nlp_agent.analyze(symptom_text=symptom_text)

        decision = self.decision_engine.evaluate(vision_output=v_output, nlp_output=n_output, mode=mode)
        self.history.append(decision)
        return decision

    def run_ablation_comparison(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None = None,
        symptom_text: str | None = None,
        vision_override: AgentOutputDict | None = None,
        nlp_override: AgentOutputDict | None = None,
    ) -> dict[OperationMode, DecisionResult]:
        """Compara (a) Visão isolada, (b) PLN isolado e (c) Sistema Integrado, reutilizando as inferências."""
        if vision_override is None and image_input is not None:
            vision_override = self.ml_agent.classify_processed_image(
                self.vision_agent.extract_and_prepare(image_input)
            )
        if nlp_override is None and symptom_text and str(symptom_text).strip():
            nlp_override = self.nlp_agent.analyze(symptom_text=symptom_text)
        return {
            "vision_only": self.diagnose(mode="vision_only", vision_override=vision_override),
            "nlp_only": self.diagnose(mode="nlp_only", nlp_override=nlp_override),
            "integrated": self.diagnose(mode="integrated", vision_override=vision_override, nlp_override=nlp_override),
        }

    # --------------------------------------------------------- imagem + pergunta
    def answer(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None,
        user_message: str,
    ) -> QAResponse:
        """Fluxo completo do PDF: imagem + mensagem -> visão, PLN (sintomas e intenção) e decisão -> resposta."""
        parts = split_user_message(user_message)
        if parts.question:
            intent, intent_conf = self.intent_classifier.predict(parts.question)
        else:
            intent, intent_conf = "diagnose", 0.0  # sem pergunta explícita: devolve o diagnóstico
        decision = self.diagnose(image_input=image_input, symptom_text=parts.symptom_text or None)
        return QAResponse(
            answer=self._compose_answer(decision, intent, user_message),
            intent=intent,
            intent_confidence=intent_conf,
            question=parts.question,
            symptom_text=parts.symptom_text,
            decision=decision,
        )

    @staticmethod
    def _compose_answer(decision: DecisionResult, intent: str, user_message: str) -> str:
        cls_name = decision.final_class
        if decision.status in (DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE, DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA) or not cls_name:
            return (
                "Não há evidência suficiente para responder com segurança. "
                + " ".join(decision.warnings[:2])
                + " Envie uma foto nítida e descreva os sintomas observados."
            ).strip()
        meta = get_disease_metadata(cls_name)
        qa = CLASS_QA_INFO.get(cls_name, {})
        who = f"{meta['common_name']} ({meta['crop']})"
        head = f"Diagnóstico mais provável: {who}, confiança {decision.combined_confidence:.0%} [{decision.status.value}]."
        if decision.status == DecisionStatus.DIVERGENCIA_DETECTADA:
            head += " ATENÇÃO: imagem e texto divergem; confirme em campo antes de agir."
        if intent == "treatment":
            body = meta["recommendation"]
        elif intent == "prevention":
            body = qa.get("prevention", meta["recommendation"])
        elif intent == "severity":
            body = qa.get("severity", "Gravidade não catalogada.")
        elif intent == "confirm":
            body = f"Com base nas evidências disponíveis, o resultado indica {who}."
            if cls_name.endswith("healthy"):
                body = "A folha aparenta estar saudável segundo as evidências disponíveis."
        else:
            body = f"Agente etiológico: {meta['pathogen']}."
        return f"{head} {body}"

    # --------------------------------------------------------------- utilidades
    def export_experimental_history(self, output_path: str | Path) -> Path:
        """Atende à Funcionalidade F08: registra o histórico experimental de decisões em JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.to_dict() for item in self.history]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def generate_report(self, decision: DecisionResult, title: str | None = None) -> str:
        return ReportGenerator.format_terminal_report(decision, title=title)

    def generate_structured_report(self, decision: DecisionResult) -> dict[str, Any]:
        return ReportGenerator.to_structured_dict(decision)
