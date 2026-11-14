"""Processamento e normalização de descrições textuais de sintomas em plantas.

Realiza:
1. Limpeza e normalização Unicode (remoção de acentos, pontuação e espaços extras).
2. Remoção seletiva de stopwords em português preservando termos de negação
   ('sem', 'nao', 'ausencia', 'livre') cruciais para diferenciar plantas sadias de doentes.
3. Extração estruturada de palavras-chave de sintomas agronômicos e cultura mencionada,
   sem jamais inventar sintomas não relatados pelo usuário.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

PORTUGUESE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
        "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
        "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
        "era", "eram", "essa", "essas", "esse", "esses", "esta", "estao", "estar",
        "estas", "este", "estes", "eu", "foi", "foram", "ha", "isso", "isto", "ja",
        "la", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha",
        "minhas", "muito", "na", "nas", "nem", "no", "nos", "nossa", "nossas",
        "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas",
        "pelo", "pelos", "por", "qual", "quando", "que", "quem", "sao", "se", "seu",
        "seus", "so", "sua", "suas", "tambem", "te", "tem", "tendo", "tenho", "ter",
        "teu", "teus", "tu", "tua", "tuas", "um", "uma", "umas", "uns", "voce", "voces",
    }
)

# Dicionário canônico de padrões de sintomas agronômicos para extração estruturada
AGRONOMIC_SYMPTOM_LEXICON: dict[str, list[str]] = {
    "aneis_concentricos": ["anel concentrico", "aneis concentricos", "circulos concentricos", "estrias concentricas", "tipo alvo", "formato de alvo", "pinta-preta", "pinta preta", "alternariose"],
    "lesoes_encharcadas": ["encharcada", "encharcadas", "encharcado", "aquosa", "aquosas", "aspecto oleoso", "queimadura umida", "necrose umida", "requeima", "mela"],
    "mofo_esbranquicado": ["mofo branco", "mofo esbranquicado", "bolor esbranquicado", "micelio branco", "micelio esbranquicado", "penugem branca", "penugem esbranquicada", "esporulacao branca"],
    "halo_amarelado": ["halo amarelo", "halo amarelado", "halo clorotico", "bordas amarelas", "bordas amareladas", "amarelecimento", "clorose"],
    "pustulas_ferruginosas": ["pustula", "pustulas", "ferrugem", "ferruginosa", "ferruginosas", "cor de canela", "cor de tijolo", "marrom-avermelhada", "marrom-avermelhadas", "po avermelhado", "po marrom"],
    "necrose_foliar": ["necrose", "necrotica", "necroticas", "necrosado", "manchas escuras", "manchas marrons", "placas marrons", "podridao", "apodrecimento"],
    "folha_saudavel": ["saudavel", "sadia", "sadias", "sadio", "sadios", "integra", "integras", "integro", "sem manchas", "sem lesoes", "sem sintomas", "livre de patogenos", "verde uniforme", "verde intensa"],
}

CROP_KEYWORDS: dict[str, list[str]] = {
    "Tomato": ["tomate", "tomateiro", "tomates", "solanum lycopersicum"],
    "Potato": ["batata", "batateira", "batatas", "solanum tuberosum"],
    "Corn_(maize)": ["milho", "milharal", "zea mays"],
}


@dataclass(slots=True)
class ProcessedText:
    """Representação estruturada do texto após limpeza e extração de termos."""

    original_text: str
    normalized_text: str
    tokens: list[str]
    extracted_symptoms: list[str] = field(default_factory=list)
    detected_crops: list[str] = field(default_factory=list)
    is_empty: bool = False

    def to_metadata(self) -> dict[str, Any]:
        """Retorna metadados estruturados da análise textual para o contrato padrão."""
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "token_count": len(self.tokens),
            "extracted_symptoms": list(self.extracted_symptoms),
            "detected_crops": list(self.detected_crops),
            "is_empty_text": self.is_empty,
        }


class TextProcessor:
    """Processador textual para descrições de sintomas agronômicos em português."""

    @staticmethod
    def strip_accents(text: str) -> str:
        """Remove acentuação gráfica mantendo caracteres ASCII base."""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def clean_and_tokenize(self, raw_text: str) -> tuple[str, list[str]]:
        """Normaliza caixa, acentos e caracteres especiais, retornando texto limpo e tokens."""
        if not raw_text or not raw_text.strip():
            return "", []

        lowered = self.strip_accents(raw_text.lower().strip())
        # Substitui hifens por espaço para preservar termos compostos como 'pinta-preta'
        cleaned = re.sub(r"[-_/]+", " ", lowered)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        raw_tokens = cleaned.split()
        stem_map = {
            "tomateiro": "tomate",
            "tomateiros": "tomate",
            "tomates": "tomate",
            "batateira": "batata",
            "batateiras": "batata",
            "batatas": "batata",
            "milharal": "milho",
        }
        filtered_tokens = [
            stem_map.get(tok, tok)
            for tok in raw_tokens
            if tok not in PORTUGUESE_STOPWORDS and len(tok) > 1
        ]
        normalized_text = " ".join(filtered_tokens)
        return normalized_text, filtered_tokens

    def extract_symptoms(self, raw_text: str) -> list[str]:
        """Extrai categorias de sintomas explicitamente mencionadas pelo usuário."""
        if not raw_text or not raw_text.strip():
            return []

        ascii_lower = self.strip_accents(raw_text.lower())
        ascii_lower = re.sub(r"\s+", " ", ascii_lower)

        # Verifica se há negação explícita antes de termos de doença (ex: "sem manchas", "sem lesões")
        has_healthy_negation = any(
            phrase in ascii_lower
            for phrase in ("sem manchas", "sem lesoes", "sem nenhuma mancha", "ausencia total de lesoes", "sem sinais de")
        )

        found_symptoms: list[str] = []
        for symptom_tag, patterns in AGRONOMIC_SYMPTOM_LEXICON.items():
            if has_healthy_negation and symptom_tag != "folha_saudavel":
                # Não extrai 'necrose_foliar' se o usuário escreveu 'sem manchas escuras'
                continue
            for pattern in patterns:
                norm_pattern = self.strip_accents(pattern.lower())
                if norm_pattern in ascii_lower:
                    found_symptoms.append(symptom_tag)
                    break

        return found_symptoms

    def detect_crops(self, raw_text: str) -> list[str]:
        """Identifica quais culturas agrícolas foram mencionadas no relato."""
        if not raw_text or not raw_text.strip():
            return []

        ascii_lower = self.strip_accents(raw_text.lower())
        detected: list[str] = []
        for crop_code, terms in CROP_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(term)}\b", ascii_lower) for term in terms):
                detected.append(crop_code)
        return detected

    def process(self, raw_text: str | None) -> ProcessedText:
        """Executa o pipeline completo de pré-processamento e extração de entidades."""
        if raw_text is None or not str(raw_text).strip():
            return ProcessedText(
                original_text="",
                normalized_text="",
                tokens=[],
                extracted_symptoms=[],
                detected_crops=[],
                is_empty=True,
            )

        text_str = str(raw_text).strip()
        normalized_text, tokens = self.clean_and_tokenize(text_str)
        extracted_symptoms = self.extract_symptoms(text_str)
        detected_crops = self.detect_crops(text_str)

        # Enriquece a representação normalizada com marcadores canônicos extraídos
        canonical_tags = (
            [f"cultura_{c.lower()}" for c in detected_crops]
            + [f"sintoma_{s.lower()}" for s in extracted_symptoms]
            + [
                f"interacao_{c.lower()}_{s.lower()}"
                for c in detected_crops
                for s in extracted_symptoms
            ]
        )
        if canonical_tags:
            normalized_text = f"{normalized_text} {' '.join(canonical_tags)}".strip()

        return ProcessedText(
            original_text=text_str,
            normalized_text=normalized_text,
            tokens=tokens,
            extracted_symptoms=extracted_symptoms,
            detected_crops=detected_crops,
            is_empty=len(tokens) == 0,
        )
