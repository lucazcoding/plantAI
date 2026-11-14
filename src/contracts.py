"""Contratos de dados padronizados entre os módulos e agentes do PlantAI.

Este módulo define o protocolo inegociável de comunicação entre:
- Agente de Visão Computacional / ML (Fabiano)
- Agente de Processamento de Linguagem Natural (Nogueira)
- Agente Coordenador e Motor de Regras/Heurísticas (Viny Carnaúba)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict

AgentType = Literal["vision", "nlp"]
OperationMode = Literal["integrated", "vision_only", "nlp_only"]


class DecisionStatus(str, Enum):
    """Status padronizados emitidos pelo motor de regras e pelo Agente Coordenador."""

    CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS = "CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS"
    DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL = "DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL"
    DIAGNOSTICO_SOMENTE_VISUAL = "DIAGNOSTICO_SOMENTE_VISUAL"
    DIAGNOSTICO_SOMENTE_TEXTUAL = "DIAGNOSTICO_SOMENTE_TEXTUAL"
    DIVERGENCIA_DETECTADA = "DIVERGENCIA_DETECTADA"
    INCONCLUSIVO_BAIXA_CONFIANCA = "INCONCLUSIVO_BAIXA_CONFIANCA"
    QUALIDADE_IMAGEM_INSUFICIENTE = "QUALIDADE_IMAGEM_INSUFICIENTE"


class AgentOutputDict(TypedDict):
    """Contrato estrito de saída exigido para os agentes especialistas (Visão e PLN)."""

    agent_type: AgentType
    predicted_class: str          # ex: "Tomato___Early_blight"
    confidence: float             # 0.0 a 1.0
    top_k: list[tuple[str, float]]  # Top-3 classes e probabilidades
    metadata: dict[str, Any]      # ex: qualidade da imagem, sintomas extraídos, etc.


@dataclass(slots=True)
class AgentPrediction:
    """Representação tipada da predição de um agente especialista, conversível para dict."""

    agent_type: AgentType
    predicted_class: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> AgentOutputDict:
        """Retorna o dicionário no formato exato do contrato padronizado."""
        return {
            "agent_type": self.agent_type,
            "predicted_class": self.predicted_class,
            "confidence": float(round(self.confidence, 4)),
            "top_k": [(cls_name, float(round(prob, 4))) for cls_name, prob in self.top_k],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: AgentOutputDict | dict[str, Any]) -> AgentPrediction:
        """Constrói e valida uma instância de AgentPrediction a partir de um dicionário."""
        validate_agent_output(data)
        return cls(
            agent_type=data["agent_type"],
            predicted_class=str(data["predicted_class"]),
            confidence=float(data["confidence"]),
            top_k=[(str(k), float(v)) for k, v in data.get("top_k", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class DecisionResult:
    """Resultado consolidado produzido pelas regras de decisão e pelo Coordenador."""

    status: DecisionStatus
    final_class: str | None
    combined_confidence: float
    operation_mode: OperationMode
    rule_triggered: str
    evidences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    candidate_hypotheses: list[tuple[str, float, str]] = field(default_factory=list)
    vision_output: AgentOutputDict | None = None
    nlp_output: AgentOutputDict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializa a decisão consolidada para dicionário estruturado."""
        return {
            "status": self.status.value,
            "final_class": self.final_class,
            "disease_info": get_disease_metadata(self.final_class),
            "combined_confidence": round(float(self.combined_confidence), 4),
            "operation_mode": self.operation_mode,
            "rule_triggered": self.rule_triggered,
            "evidences": list(self.evidences),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "candidate_hypotheses": [
                {"class": cls_name, "confidence": round(float(conf), 4), "source": src}
                for cls_name, conf, src in self.candidate_hypotheses
            ],
            "vision_output": self.vision_output,
            "nlp_output": self.nlp_output,
        }


PLANTVILLAGE_CLASS_INFO: dict[str, dict[str, str]] = {
    "Tomato___Early_blight": {
        "crop": "Tomate (Solanum lycopersicum)",
        "common_name": "Pinta-preta (Alternariose)",
        "pathogen": "Alternaria solani",
        "recommendation": (
            "Remover folhas baixeiras severamente atacadas, evitar irrigação por aspersão "
            "no final da tarde, melhorar a ventilação entre plantas e consultar um "
            "engenheiro agrônomo para manejo fungicida registrado."
        ),
    },
    "Tomato___Late_blight": {
        "crop": "Tomate (Solanum lycopersicum)",
        "common_name": "Requeima (Mela)",
        "pathogen": "Phytophthora infestans",
        "recommendation": (
            "Doença de rápida progressão em condições frias e úmidas. Isolar focos iniciais, "
            "reduzir molhamento foliar prolongado e buscar assistência agronômica imediata."
        ),
    },
    "Tomato___Healthy": {
        "crop": "Tomate (Solanum lycopersicum)",
        "common_name": "Planta Saudável",
        "pathogen": "Nenhum patógeno detectado",
        "recommendation": (
            "Manter o monitoramento fitossanitário periódico, adubação equilibrada e "
            "irrigação adequada sem encharcamento."
        ),
    },
    "Potato___Early_blight": {
        "crop": "Batata (Solanum tuberosum)",
        "common_name": "Pinta-preta da Batateira",
        "pathogen": "Alternaria solani",
        "recommendation": (
            "Realizar rotação de culturas, manter nutrição equilibrada (especialmente potássio "
            "e nitrogênio), eliminar restos culturais e avaliar manejo técnico especializado."
        ),
    },
    "Potato___Late_blight": {
        "crop": "Batata (Solanum tuberosum)",
        "common_name": "Requeima da Batateira",
        "pathogen": "Phytophthora infestans",
        "recommendation": (
            "Alto potencial destrutivo. Monitorar umidade relativa e temperatura, utilizar "
            "tubérculos-semente sadios e acionar avaliação agronômica urgente."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Milho (Zea mays)",
        "common_name": "Ferrugem-comum do Milho",
        "pathogen": "Puccinia sorghi",
        "recommendation": (
            "Priorizar híbridos resistentes em plantios futuros, monitorar a evolução das "
            "pústulas antes do pendoamento e consultar recomendação agronômica local."
        ),
    },
}


def get_disease_metadata(class_name: str | None) -> dict[str, str]:
    """Retorna metadados legíveis sobre uma classe do PlantVillage."""
    if not class_name:
        return {
            "crop": "Não determinado",
            "common_name": "Diagnóstico inconclusivo / requer revisão",
            "pathogen": "Não determinado",
            "recommendation": (
                "Forneça uma nova fotografia nítida da folha sob iluminação natural e/ou "
                "detalhe os sintomas observados na planta para nova análise."
            ),
        }
    return PLANTVILLAGE_CLASS_INFO.get(
        class_name,
        {
            "crop": class_name.split("___")[0] if "___" in class_name else "Cultura agrícola",
            "common_name": class_name.replace("___", " - ").replace("_", " ").strip(),
            "pathogen": "Consultar catálogo fitossanitário PlantVillage",
            "recommendation": (
                "Confirmar o diagnóstico visual em campo com auxílio de profissional de agronomia."
            ),
        },
    )


def validate_agent_output(payload: AgentOutputDict | dict[str, Any]) -> AgentOutputDict:
    """Valida se o dicionário retornado por um agente obedece ao contrato padronizado."""
    required_keys = {"agent_type", "predicted_class", "confidence", "top_k", "metadata"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Contrato violado: chaves obrigatórias ausentes: {sorted(missing)}")

    if payload["agent_type"] not in ("vision", "nlp"):
        raise ValueError(f"agent_type inválido: {payload['agent_type']!r}. Esperado 'vision' ou 'nlp'.")

    confidence = float(payload["confidence"])
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence fora do intervalo [0.0, 1.0]: {confidence}")

    if not isinstance(payload["top_k"], list):
        raise ValueError("top_k deve ser uma lista de tuplas (classe, probabilidade).")

    if not isinstance(payload["metadata"], dict):
        raise ValueError("metadata deve ser um dicionário.")

    return {
        "agent_type": payload["agent_type"],
        "predicted_class": str(payload["predicted_class"]),
        "confidence": confidence,
        "top_k": [(str(cls_name), float(prob)) for cls_name, prob in payload["top_k"]],
        "metadata": dict(payload["metadata"]),
    }

