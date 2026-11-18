"""Formatador de relatórios técnicos e científicos do sistema multiagente PlantAI.

Atende ao requisito funcional F07:
- Classe identificada e nome comum da doença / patógeno.
- Confiança consolidada e modo de operação (Estudo de Ablação).
- Informações estruturadas extraídas da descrição textual (PLN).
- Qualidade da imagem e Top-K de cada agente especialista.
- Evidências utilizadas e regra acionada.
- Avisos sobre incertezas/divergências e recomendações gerais.
"""

from __future__ import annotations

from typing import Any

from src.contracts import DecisionResult, get_disease_metadata


class ReportGenerator:
    """Gera relatórios estruturados (dict/markdown/texto de terminal) a partir de DecisionResult."""

    DISCLAIMER = (
        "AVISO CIENTÍFICO E AGRONÔMICO: Este resultado corresponde a uma predição "
        "computacional experimental e NÃO substitui o diagnóstico presencial de um "
        "engenheiro agrônomo."
    )

    @classmethod
    def to_structured_dict(cls, decision: DecisionResult) -> dict[str, Any]:
        """Retorna um dicionário completo com todas as seções do relatório."""
        base = decision.to_dict()
        base["disclaimer"] = cls.DISCLAIMER
        return base

    @classmethod
    def format_terminal_report(cls, decision: DecisionResult, title: str | None = None) -> str:
        """Formata o diagnóstico consolidado para exibição clara no terminal ou notebook."""
        disease_meta = get_disease_metadata(decision.final_class)
        sep = "=" * 78
        subsep = "-" * 78

        lines: list[str] = [sep]
        if title:
            lines.append(f"  PLANTAI - RELATÓRIO DE DIAGNÓSTICO MULTIAGENTE | {title}")
        else:
            lines.append("  PLANTAI - RELATÓRIO DE DIAGNÓSTICO MULTIAGENTE")
        lines.append(sep)

        lines.append(f"Status da Decisão    : {decision.status.value}")
        lines.append(f"Regra Acionada       : {decision.rule_triggered}")
        lines.append(f"Modo de Operação     : {decision.operation_mode}")
        lines.append(
            f"Classe Identificada  : {decision.final_class or 'INCONCLUSIVO / REQUER REVISÃO'}"
        )
        lines.append(f"Cultura / Doença     : {disease_meta['crop']} — {disease_meta['common_name']}")
        lines.append(f"Agente Etiológico    : {disease_meta['pathogen']}")
        lines.append(f"Confiança Consolidada: {decision.combined_confidence:.2%}")
        lines.append(subsep)

        # Resumo dos Agentes Especialistas
        lines.append("[1] SAÍDAS DOS AGENTES ESPECIALISTAS (CONTRATO PADRONIZADO):")
        if decision.vision_output is not None:
            v = decision.vision_output
            v_meta = v.get("metadata", {})
            v_topk_str = ", ".join(f"{c} ({p:.1%})" for c, p in v.get("top_k", []))
            lines.append(
                f"  • VisionAgent (MobileNetV2): Top-1='{v['predicted_class']}' "
                f"(conf={v['confidence']:.2%}) | Top-K=[{v_topk_str}]"
            )
            if "brightness" in v_meta and "blur_score" in v_meta:
                lines.append(
                    f"    Qualidade Visual: válida={v_meta.get('image_valid')} | "
                    f"brilho={v_meta.get('brightness')} | nitidez={v_meta.get('blur_score')}"
                )
        else:
            lines.append("  • VisionAgent (MobileNetV2): Não acionado neste modo.")

        if decision.nlp_output is not None:
            n = decision.nlp_output
            n_meta = n.get("metadata", {})
            n_topk_str = ", ".join(f"{c} ({p:.1%})" for c, p in n.get("top_k", []))
            lines.append(
                f"  • NLPAgent (TF-IDF + ML)   : Top-1='{n['predicted_class']}' "
                f"(conf={n['confidence']:.2%}) | Top-K=[{n_topk_str}]"
            )
            symptoms = n_meta.get("extracted_symptoms", [])
            crops = n_meta.get("detected_crops", [])
            lines.append(
                f"    Sintomas Extraídos: {symptoms if symptoms else 'nenhum termo canônico isolado'} "
                f"| Cultura citada: {crops if crops else 'não especificada'}"
            )
        else:
            lines.append("  • NLPAgent (TF-IDF + ML)   : Sem descrição textual informada.")

        # Hipóteses candidatas (útil especialmente em divergências)
        if decision.candidate_hypotheses:
            lines.append(subsep)
            lines.append("[2] HIPÓTESES CONSIDERADAS PELO COORDENADOR:")
            for cls_name, conf, src in decision.candidate_hypotheses:
                lines.append(f"  - {cls_name} (confiança={conf:.2%}, origem={src})")

        # Evidências
        lines.append(subsep)
        lines.append("[3] EVIDÊNCIAS UTILIZADAS NA TOMADA DE DECISÃO:")
        for ev in decision.evidences:
            lines.append(f"  ✓ {ev}")

        # Alertas / Incertezas
        if decision.warnings:
            lines.append(subsep)
            lines.append("[4] ALERTAS DE INCERTEZA E DIVERGÊNCIA:")
            for warn in decision.warnings:
                lines.append(f"  ! {warn}")

        # Recomendações
        lines.append(subsep)
        lines.append("[5] RECOMENDAÇÕES TÉCNICAS GERAIS:")
        for rec in decision.recommendations:
            lines.append(f"  -> {rec}")

        lines.append(subsep)
        lines.append(f"  {cls.DISCLAIMER}")
        lines.append(sep)
        return "\n".join(lines)

