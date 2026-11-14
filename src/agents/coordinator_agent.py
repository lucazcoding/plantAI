"""Agente 4 — Coordenador Central do Sistema Multiagente PlantAI (`CoordinatorAgent`).

Atende à Seção 4.5 e Funcionalidade F08 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
1. Receber a solicitação do usuário (imagem e/ou descrição textual).
2. Verificar quais entradas foram fornecidas.
3. Acionar os módulos necessários (`VisionAgent` -> `MLAgent`, e `NLPAgent`).
4. Receber os resultados no contrato padronizado.
5. Comparar as informações e aplicar regras e heurísticas (`DecisionRulesEngine`).
6. Identificar divergências e determinar se há informações suficientes.
7. Produzir o relatório final (`ReportGenerator`) e registrar o histórico experimental (F08).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.agents.ml_agent import MLAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.vision_agent import VisionAgent
from src.contracts import (
    AgentOutputDict,
    DecisionResult,
    OperationMode,
    validate_agent_output,
)
from src.decision.decision_rules import DecisionRulesEngine
from src.reports.report_generator import ReportGenerator


class CoordinatorAgent:
    """Agente Coordenador responsável pela fusão de evidências, resolução de conflitos e ablação."""

    def __init__(
        self,
        vision_agent: VisionAgent | None = None,
        ml_agent: MLAgent | None = None,
        nlp_agent: NLPAgent | None = None,
        decision_engine: DecisionRulesEngine | None = None,
    ) -> None:
        self.vision_agent = vision_agent or VisionAgent(ml_agent=ml_agent)
        self.ml_agent = ml_agent or self.vision_agent.ml_agent
        self.nlp_agent = nlp_agent or NLPAgent()
        self.decision_engine = decision_engine or DecisionRulesEngine()
        self.history: list[DecisionResult] = []

    def diagnose(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None = None,
        symptom_text: str | None = None,
        mode: OperationMode | None = None,
        vision_override: AgentOutputDict | None = None,
        nlp_override: AgentOutputDict | None = None,
        mock_vision_probs: dict[str, float] | None = None,
        mock_nlp_probs: dict[str, float] | None = None,
    ) -> DecisionResult:
        """Executa o fluxo completo de coordenação multiagente e retorna a decisão consolidada."""
        v_output: AgentOutputDict | None = None
        n_output: AgentOutputDict | None = None

        run_vision = mode in (None, "integrated", "vision_only")
        run_nlp = mode in (None, "integrated", "nlp_only")

        if run_vision:
            if vision_override is not None:
                v_output = validate_agent_output(vision_override)
            elif image_input is not None or mock_vision_probs is not None:
                effective_img = (
                    image_input
                    if image_input is not None
                    else self.vision_agent.image_processor.create_synthetic_leaf_image("healthy_sharp")
                )
                processed_img = self.vision_agent.extract_and_prepare(effective_img)
                v_output = self.ml_agent.classify_processed_image(
                    processed_image=processed_img,
                    mock_probabilities=mock_vision_probs,
                )

        if run_nlp:
            if nlp_override is not None:
                n_output = validate_agent_output(nlp_override)
            elif (symptom_text is not None and str(symptom_text).strip() != "") or mock_nlp_probs is not None:
                n_output = self.nlp_agent.analyze(
                    symptom_text=symptom_text,
                    mock_probabilities=mock_nlp_probs,
                )

        decision = self.decision_engine.evaluate(
            vision_output=v_output,
            nlp_output=n_output,
            mode=mode,
        )
        self.history.append(decision)
        return decision

    def run_ablation_comparison(
        self,
        image_input: str | Path | bytes | Image.Image | np.ndarray | None = None,
        symptom_text: str | None = None,
        vision_override: AgentOutputDict | None = None,
        nlp_override: AgentOutputDict | None = None,
        mock_vision_probs: dict[str, float] | None = None,
        mock_nlp_probs: dict[str, float] | None = None,
    ) -> dict[OperationMode, DecisionResult]:
        """Compara cientificamente (a) Visão isolada, (b) PLN isolado e (c) Sistema Integrado."""
        vision_only_res = self.diagnose(
            image_input=image_input,
            symptom_text=None,
            mode="vision_only",
            vision_override=vision_override,
            mock_vision_probs=mock_vision_probs,
        )
        nlp_only_res = self.diagnose(
            image_input=None,
            symptom_text=symptom_text,
            mode="nlp_only",
            nlp_override=nlp_override,
            mock_nlp_probs=mock_nlp_probs,
        )
        integrated_res = self.diagnose(
            image_input=image_input,
            symptom_text=symptom_text,
            mode="integrated",
            vision_override=vision_override,
            nlp_override=nlp_override,
            mock_vision_probs=mock_vision_probs,
            mock_nlp_probs=mock_nlp_probs,
        )
        return {
            "vision_only": vision_only_res,
            "nlp_only": nlp_only_res,
            "integrated": integrated_res,
        }

    def export_experimental_history(self, output_path: str | Path) -> Path:
        """Atende à Funcionalidade F08: registra o histórico experimental de decisões em JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.to_dict() for item in self.history]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def generate_report(self, decision: DecisionResult, title: str | None = None) -> str:
        """Formata um relatório textual completo a partir de uma decisão do Coordenador."""
        return ReportGenerator.format_terminal_report(decision, title=title)

    def generate_structured_report(self, decision: DecisionResult) -> dict[str, Any]:
        """Retorna o relatório em formato de dicionário estruturado."""
        return ReportGenerator.to_structured_dict(decision)
