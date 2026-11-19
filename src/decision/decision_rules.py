"""Regras explícitas e heurísticas para fusão de decisão e resolução de conflitos no PlantAI.

Implementa as 6 heurísticas obrigatórias da arquitetura multiagente:
1. Validação de Qualidade Visual (rejeição/alerta crítico para imagens corrompidas, escuras ou borradas).
2. Modo Somente Visual / Fallback (e Modo Somente PLN para estudo de ablação).
3. Concordância com Boost de Confiança (`CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS`).
4. Desambiguação Visual pelo PLN quando a imagem tem confiança moderada (`0.45` a `0.75`)
   e está em dúvida entre classes do `top_k` (`DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL`).
5. Divergência / Conflito Explícito quando Visão e PLN discordam (`DIVERGENCIA_DETECTADA`).
6. Incerteza Geral quando ambos os agentes apresentam baixa confiança (`INCONCLUSIVO_BAIXA_CONFIANCA`).
"""

from __future__ import annotations

from typing import Any

from src.contracts import (
    AgentOutputDict,
    DecisionResult,
    DecisionStatus,
    OperationMode,
    get_disease_metadata,
    validate_agent_output,
)

DEFAULT_VISION_CONFIDENCE_THRESHOLD: float = 0.75
DEFAULT_NLP_HIGH_CONFIDENCE_THRESHOLD: float = 0.70
DEFAULT_MIN_CONFIDENCE_THRESHOLD: float = 0.45
DEFAULT_TOP2_PROXIMITY_GAP: float = 0.25


class DecisionRulesEngine:
    """Motor de regras e heurísticas determinístico e auditável para fusão multiagente."""

    def __init__(
        self,
        vision_confidence_threshold: float = DEFAULT_VISION_CONFIDENCE_THRESHOLD,
        nlp_high_confidence_threshold: float = DEFAULT_NLP_HIGH_CONFIDENCE_THRESHOLD,
        min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD,
        top2_proximity_gap: float = DEFAULT_TOP2_PROXIMITY_GAP,
        vision_weight: float = 0.60,
        nlp_weight: float = 0.40,
    ) -> None:
        self.vision_confidence_threshold = vision_confidence_threshold
        self.nlp_high_confidence_threshold = nlp_high_confidence_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.top2_proximity_gap = top2_proximity_gap
        self.vision_weight = vision_weight
        self.nlp_weight = nlp_weight

    @staticmethod
    def compute_concordance_boost(conf_vision: float, conf_nlp: float) -> float:
        """Calcula a confiança combinada com boost bayesiano moderado quando há concordância.

        A média ponderada base (60% visão + 40% PLN) recebe um incremento proporcional
        à concordância entre evidências independentes, limitado a 0.99.
        """
        weighted_avg = 0.60 * conf_vision + 0.40 * conf_nlp
        # Reforço de evidências independentes (noisy-OR suavizado)
        independent_reinforcement = 1.0 - (1.0 - conf_vision) * (1.0 - conf_nlp)
        boosted = 0.65 * weighted_avg + 0.35 * independent_reinforcement
        return float(min(0.99, round(boosted, 4)))

    @staticmethod
    def is_image_quality_acceptable(vision_output: AgentOutputDict) -> tuple[bool, list[str]]:
        """Verifica nos metadados do agente visual se a imagem passou nos critérios mínimos."""
        meta: dict[str, Any] = vision_output.get("metadata", {})
        issues: list[str] = list(meta.get("quality_issues", []))

        if meta.get("is_corrupted", False):
            if not issues:
                issues.append("Imagem corrompida ou ilegível.")
            return False, issues

        if meta.get("is_too_dark", False):
            if not any("escura" in msg.lower() for msg in issues):
                issues.append("Imagem extremamente escura.")
            return False, issues

        if meta.get("is_blurry", False):
            if not any("borrada" in msg.lower() for msg in issues):
                issues.append("Imagem borrada (nitidez insuficiente).")
            return False, issues

        if meta.get("image_valid") is False:
            if not issues:
                issues.append("Imagem reprovada na validação de qualidade visual.")
            return False, issues

        return True, issues

    def evaluate(
        self,
        vision_output: AgentOutputDict | None = None,
        nlp_output: AgentOutputDict | None = None,
        mode: OperationMode | None = None,
    ) -> DecisionResult:
        """Aplica a cadeia de regras de decisão sobre as saídas dos agentes especialistas."""
        v_out = validate_agent_output(vision_output) if vision_output is not None else None
        n_out = validate_agent_output(nlp_output) if nlp_output is not None else None

        # Determina se o PLN está efetivamente vazio
        if n_out is not None and (
            n_out["metadata"].get("is_empty_text", False)
            or n_out["predicted_class"] == "UNKNOWN"
            or n_out["confidence"] == 0.0
        ):
            n_out = None

        # Determina o modo de operação efetivo
        effective_mode: OperationMode
        if mode is not None:
            effective_mode = mode
        elif v_out is not None and n_out is not None:
            effective_mode = "integrated"
        elif v_out is not None:
            effective_mode = "vision_only"
        else:
            effective_mode = "nlp_only"

        # ---------------------------------------------------------------------
        # REGRA 1: Validação de Qualidade Visual (se houver entrada de visão)
        # ---------------------------------------------------------------------
        if v_out is not None and effective_mode in ("integrated", "vision_only"):
            is_valid_img, quality_issues = self.is_image_quality_acceptable(v_out)
            if not is_valid_img:
                warnings = [
                    "ALERTA CRÍTICO DE QUALIDADE VISUAL: A fotografia enviada não atende aos "
                    "critérios mínimos para inferência confiável.",
                    *quality_issues,
                ]
                evidences = [
                    f"Inspeção visual reprovada: {'; '.join(quality_issues)}",
                ]
                if n_out is not None:
                    evidences.append(
                        f"Relato textual apontou hipótese preliminar '{n_out['predicted_class']}' "
                        f"(confiança={n_out['confidence']:.2%}), mas o diagnóstico visual foi bloqueado."
                    )
                return DecisionResult(
                    status=DecisionStatus.QUALIDADE_IMAGEM_INSUFICIENTE,
                    final_class=None,
                    combined_confidence=0.0,
                    operation_mode=effective_mode,
                    rule_triggered="R01_IMAGE_QUALITY_REJECTION",
                    evidences=evidences,
                    warnings=warnings,
                    recommendations=[
                        "Capture uma nova fotografia da folha bem iluminada (luz natural indireta), "
                        "com foco nítido sobre as lesões e fundo contrastante."
                    ],
                    candidate_hypotheses=(
                        [(n_out["predicted_class"], n_out["confidence"], "nlp")] if n_out else []
                    ),
                    vision_output=v_out,
                    nlp_output=n_out,
                )

        # ---------------------------------------------------------------------
        # REGRA 2A: Modo Somente Visual (Fallback quando texto não é enviado ou ablação)
        # ---------------------------------------------------------------------
        if effective_mode == "vision_only" or (v_out is not None and n_out is None):
            if v_out is None:
                return self._build_inconclusive_no_inputs()

            v_class = v_out["predicted_class"]
            v_conf = v_out["confidence"]

            if v_conf >= self.vision_confidence_threshold:
                info = get_disease_metadata(v_class)
                return DecisionResult(
                    status=DecisionStatus.DIAGNOSTICO_SOMENTE_VISUAL,
                    final_class=v_class,
                    combined_confidence=v_conf,
                    operation_mode="vision_only",
                    rule_triggered="R02_VISION_ONLY_HIGH_CONFIDENCE",
                    evidences=[
                        f"Modo Somente Visual: MobileNetV2 classificou '{v_class}' com "
                        f"confiança {v_conf:.2%} (>= limiar {self.vision_confidence_threshold:.2%}).",
                        "Nenhuma descrição textual de sintomas foi fornecida pelo usuário.",
                    ],
                    warnings=[
                        "Análise baseada exclusivamente na imagem. A adição de descrição textual dos "
                        "sintomas permite validação cruzada pelo módulo de PLN."
                    ],
                    recommendations=[info["recommendation"]],
                    candidate_hypotheses=[(v_class, v_conf, "vision")],
                    vision_output=v_out,
                    nlp_output=None,
                )

            return DecisionResult(
                status=DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA,
                final_class=None,
                combined_confidence=v_conf,
                operation_mode="vision_only",
                rule_triggered="R02_VISION_ONLY_LOW_CONFIDENCE",
                evidences=[
                    f"Modo Somente Visual: MobileNetV2 obteve apenas {v_conf:.2%} para '{v_class}', "
                    f"abaixo do limiar de segurança visual ({self.vision_confidence_threshold:.2%})."
                ],
                warnings=[
                    "Confiança visual insuficiente para diagnóstico isolado sem relato de sintomas."
                ],
                recommendations=[
                    "Descreva textualmente os sintomas observados na planta (ex: anéis concêntricos, "
                    "lesões encharcadas, pústulas) para auxiliar o desempate pelo Agente de PLN."
                ],
                candidate_hypotheses=[
                    (cls_name, prob, "vision") for cls_name, prob in v_out["top_k"][:2]
                ],
                vision_output=v_out,
                nlp_output=None,
            )

        # ---------------------------------------------------------------------
        # REGRA 2B: Modo Somente PLN (para Estudo de Ablação Científico)
        # ---------------------------------------------------------------------
        if effective_mode == "nlp_only" or (n_out is not None and v_out is None):
            if n_out is None:
                return self._build_inconclusive_no_inputs()

            n_class = n_out["predicted_class"]
            n_conf = n_out["confidence"]
            symptoms = n_out["metadata"].get("extracted_symptoms", [])

            if n_conf >= self.min_confidence_threshold:
                info = get_disease_metadata(n_class)
                return DecisionResult(
                    status=DecisionStatus.DIAGNOSTICO_SOMENTE_TEXTUAL,
                    final_class=n_class,
                    combined_confidence=n_conf,
                    operation_mode="nlp_only",
                    rule_triggered="R02B_NLP_ONLY_CLASSIFICATION",
                    evidences=[
                        f"Modo Somente PLN: Classificador textual previu '{n_class}' "
                        f"com confiança {n_conf:.2%}.",
                        f"Sintomas extraídos do relato: {symptoms if symptoms else 'termos gerais TF-IDF'}.",
                    ],
                    warnings=[
                        "Diagnóstico baseado apenas na descrição textual. Recomenda-se enviar "
                        "fotografia da folha para confirmação visual."
                    ],
                    recommendations=[info["recommendation"]],
                    candidate_hypotheses=[(n_class, n_conf, "nlp")],
                    vision_output=None,
                    nlp_output=n_out,
                )

            return DecisionResult(
                status=DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA,
                final_class=None,
                combined_confidence=n_conf,
                operation_mode="nlp_only",
                rule_triggered="R02B_NLP_ONLY_LOW_CONFIDENCE",
                evidences=[
                    f"Modo Somente PLN: Confiança textual ({n_conf:.2%}) abaixo do limiar mínimo "
                    f"({self.min_confidence_threshold:.2%})."
                ],
                warnings=["Relato textual pouco específico para classificação conclusiva."],
                recommendations=[
                    "Envie uma fotografia nítida da folha e detalhe melhor o formato e a cor das lesões."
                ],
                candidate_hypotheses=[(n_class, n_conf, "nlp")],
                vision_output=None,
                nlp_output=n_out,
            )

        # A partir daqui temos `v_out` e `n_out` válidos no modo `integrated`
        assert v_out is not None and n_out is not None
        v_class = v_out["predicted_class"]
        v_conf = v_out["confidence"]
        n_class = n_out["predicted_class"]
        n_conf = n_out["confidence"]
        symptoms = n_out["metadata"].get("extracted_symptoms", [])

        # ---------------------------------------------------------------------
        # REGRA 6: Incerteza Geral (ambos abaixo do limiar mínimo)
        # ---------------------------------------------------------------------
        if v_conf < self.min_confidence_threshold and n_conf < self.min_confidence_threshold:
            combined = round(self.vision_weight * v_conf + self.nlp_weight * n_conf, 4)
            return DecisionResult(
                status=DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA,
                final_class=None,
                combined_confidence=combined,
                operation_mode="integrated",
                rule_triggered="R06_GENERAL_UNCERTAINTY_LOW_CONFIDENCE",
                evidences=[
                    f"Agente Visual apresentou baixa confiança ({v_conf:.2%} em '{v_class}', "
                    f"< {self.min_confidence_threshold:.2%}).",
                    f"Agente de PLN também apresentou baixa confiança ({n_conf:.2%} em '{n_class}', "
                    f"< {self.min_confidence_threshold:.2%}).",
                ],
                warnings=[
                    "Tanto a análise de imagem quanto a descrição de sintomas apresentaram "
                    "elevada incerteza estatística. Nenhuma conclusão automática pode ser emitida."
                ],
                recommendations=[
                    "Forneça uma nova fotografia em melhor ângulo/iluminação e descreva com mais "
                    "detalhes os sintomas visíveis nas folhas e no caule."
                ],
                candidate_hypotheses=[
                    (v_class, v_conf, "vision"),
                    (n_class, n_conf, "nlp"),
                ],
                vision_output=v_out,
                nlp_output=n_out,
            )

        # ---------------------------------------------------------------------
        # REGRA 3: Concordância Direta na Top-1 (Boost de Confiança)
        # ---------------------------------------------------------------------
        if v_class == n_class:
            boosted_conf = self.compute_concordance_boost(v_conf, n_conf)
            info = get_disease_metadata(v_class)
            return DecisionResult(
                status=DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS,
                final_class=v_class,
                combined_confidence=boosted_conf,
                operation_mode="integrated",
                rule_triggered="R03_EXACT_CONCORDANCE_BOOST",
                evidences=[
                    f"Concordância exata entre agentes: Visão ('{v_class}', {v_conf:.2%}) e "
                    f"PLN ('{n_class}', {n_conf:.2%}) convergiram para o mesmo diagnóstico.",
                    f"Confiança consolidada elevada para {boosted_conf:.2%} por fusão de múltiplas evidências.",
                    f"Sintomas textuais identificados: {symptoms if symptoms else 'compatíveis via TF-IDF'}.",
                ],
                warnings=[],
                recommendations=[info["recommendation"]],
                candidate_hypotheses=[(v_class, boosted_conf, "vision+nlp")],
                vision_output=v_out,
                nlp_output=n_out,
            )

        # ---------------------------------------------------------------------
        # REGRA 4: Desambiguação Visual pelo PLN (Visão moderada/em dúvida no top-k)
        # ---------------------------------------------------------------------
        vision_top_dict = dict(v_out["top_k"][:3])
        is_vision_moderate = self.min_confidence_threshold <= v_conf < self.vision_confidence_threshold

        # Caso 4A: Visão está com confiança moderada (ex: 0.45 a 0.75), em dúvida no top_k,
        # e o PLN aponta com alta confiança (>= 0.70) para uma das classes presentes no top_k visual
        if (
            is_vision_moderate
            and n_class in vision_top_dict
            and n_conf >= self.nlp_high_confidence_threshold
            and v_conf - vision_top_dict[n_class] <= self.top2_proximity_gap
            and 0.45 * vision_top_dict[n_class] + 0.55 * n_conf >= self.min_confidence_threshold
        ):
            v_prob_for_nlp_class = vision_top_dict[n_class]
            # Fusão ponderada priorizando a confirmação cruzada entre a hipótese visual secundária e o PLN forte
            disambiguated_conf = round(0.45 * v_prob_for_nlp_class + 0.55 * n_conf, 4)
            info = get_disease_metadata(n_class)
            return DecisionResult(
                status=DecisionStatus.DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL,
                final_class=n_class,
                combined_confidence=disambiguated_conf,
                operation_mode="integrated",
                rule_triggered="R04_NLP_DISAMBIGUATES_MODERATE_VISION",
                evidences=[
                    f"Agente Visual estava em dúvida moderada entre '{v_class}' ({v_conf:.2%}) "
                    f"e '{n_class}' ({v_prob_for_nlp_class:.2%}) no seu Top-K.",
                    f"Agente de PLN identificou '{n_class}' com alta confiança ({n_conf:.2%}) "
                    f"a partir dos sintomas descritos ({symptoms}), desempatando a ambiguidade visual.",
                ],
                warnings=[
                    f"Atenção: O diagnóstico visual isolado sugeria '{v_class}' com margem estreita, "
                    f"sendo desempatado para '{n_class}' pela forte evidência textual dos sintomas."
                ],
                recommendations=[info["recommendation"]],
                candidate_hypotheses=[
                    (n_class, disambiguated_conf, "nlp_disambiguated"),
                    (v_class, v_conf, "vision_top1"),
                ],
                vision_output=v_out,
                nlp_output=n_out,
            )

        # ---------------------------------------------------------------------
        # REGRA 3B: Concordância Cruzada Próxima no Top-2 (ambos confiantes e próximos)
        # ---------------------------------------------------------------------
        nlp_top2_dict = dict(n_out["top_k"][:2])
        vision_top2_dict = dict(v_out["top_k"][:2])

        if (
            v_conf >= self.vision_confidence_threshold
            and v_class in nlp_top2_dict
            and (n_conf - nlp_top2_dict[v_class]) <= self.top2_proximity_gap
        ):
            boosted_conf = self.compute_concordance_boost(v_conf, nlp_top2_dict[v_class])
            info = get_disease_metadata(v_class)
            return DecisionResult(
                status=DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS,
                final_class=v_class,
                combined_confidence=boosted_conf,
                operation_mode="integrated",
                rule_triggered="R03B_TOP2_CROSS_CONCORDANCE",
                evidences=[
                    f"Agente Visual classificou '{v_class}' com alta confiança ({v_conf:.2%}).",
                    f"Agente de PLN incluiu '{v_class}' entre as 2 principais hipóteses "
                    f"({nlp_top2_dict[v_class]:.2%}, próxima à top-1 '{n_class}' com {n_conf:.2%}).",
                ],
                warnings=[],
                recommendations=[info["recommendation"]],
                candidate_hypotheses=[
                    (v_class, boosted_conf, "vision+nlp_top2"),
                    (n_class, n_conf, "nlp_top1"),
                ],
                vision_output=v_out,
                nlp_output=n_out,
            )

        if (
            n_conf >= self.nlp_high_confidence_threshold
            and n_class in vision_top2_dict
            and (v_conf - vision_top2_dict[n_class]) <= self.top2_proximity_gap
        ):
            boosted_conf = self.compute_concordance_boost(vision_top2_dict[n_class], n_conf)
            info = get_disease_metadata(n_class)
            return DecisionResult(
                status=DecisionStatus.CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS,
                final_class=n_class,
                combined_confidence=boosted_conf,
                operation_mode="integrated",
                rule_triggered="R03B_TOP2_CROSS_CONCORDANCE",
                evidences=[
                    f"Agente de PLN classificou '{n_class}' com alta confiança ({n_conf:.2%}).",
                    f"Agente Visual incluiu '{n_class}' na sua Top-2 com probabilidade próxima "
                    f"({vision_top2_dict[n_class]:.2%} vs Top-1 '{v_class}' {v_conf:.2%}).",
                ],
                warnings=[],
                recommendations=[info["recommendation"]],
                candidate_hypotheses=[
                    (n_class, boosted_conf, "nlp+vision_top2"),
                    (v_class, v_conf, "vision_top1"),
                ],
                vision_output=v_out,
                nlp_output=n_out,
            )

        # ---------------------------------------------------------------------
        # REGRA 5: Divergência / Conflito Explícito entre Visão e PLN
        # ---------------------------------------------------------------------
        conflict_confidence = round(max(v_conf, n_conf), 4)
        return DecisionResult(
            status=DecisionStatus.DIVERGENCIA_DETECTADA,
            final_class=None,
            combined_confidence=conflict_confidence,
            operation_mode="integrated",
            rule_triggered="R05_EXPLICIT_AGENT_DIVERGENCE",
            evidences=[
                f"Conflito explícito entre especialistas: Agente Visual previu '{v_class}' "
                f"(confiança={v_conf:.2%}), enquanto Agente de PLN previu '{n_class}' "
                f"(confiança={n_conf:.2%}).",
                f"Sintomas textuais extraídos ({symptoms}) divergem do padrão visual dominante na imagem.",
            ],
            warnings=[
                "DIVERGÊNCIA DETECTADA: Os modelos de Visão Computacional e PLN apontam para "
                "diagnósticos conflitantes. Por rigor científico, o sistema NÃO oculta essa "
                "contradição e não emite um laudo único automático."
            ],
            recommendations=[
                "1. Verifique se a fotografia enviada corresponde à mesma folha cujos sintomas foram descritos.",
                "2. Envie uma nova foto focando na face inferior e superior da folha.",
                "3. Solicite revisão técnica presencial de um engenheiro agrônomo considerando ambas as hipóteses.",
            ],
            candidate_hypotheses=[
                (v_class, v_conf, "vision"),
                (n_class, n_conf, "nlp"),
            ],
            vision_output=v_out,
            nlp_output=n_out,
        )

    def _build_inconclusive_no_inputs(self) -> DecisionResult:
        """Retorno seguro quando nem imagem nem texto válido foram fornecidos."""
        return DecisionResult(
            status=DecisionStatus.INCONCLUSIVO_BAIXA_CONFIANCA,
            final_class=None,
            combined_confidence=0.0,
            operation_mode="integrated",
            rule_triggered="R00_NO_VALID_INPUTS",
            evidences=["Nenhuma imagem válida ou descrição textual foi fornecida para análise."],
            warnings=["Entrada vazia: impossível realizar inferência."],
            recommendations=["Forneça pelo menos uma fotografia de folha ou uma descrição dos sintomas."],
            candidate_hypotheses=[],
            vision_output=None,
            nlp_output=None,
        )

