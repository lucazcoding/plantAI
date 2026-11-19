"""Testes de integridade dos CSVs de apoio em data/.

Validam estrutura, preenchimento, IDs e rótulos sem carregar modelos
ou depender de rede.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SYMPTOMS_CSV = DATA / "symptoms_sample.csv"
INTENTS_CSV = DATA / "question_intents.csv"

SYMPTOMS_COLUMNS = ["id", "crop", "disease_class", "symptom_description"]
INTENTS_COLUMNS = ["id", "intent", "question"]

CROP_CLASS_PREFIX = {
    "Tomate": "Tomato___",
    "Batata": "Potato___",
    "Milho": "Corn_(maize)___",
}
ALLOWED_INTENTS = {"diagnose", "treatment", "prevention", "severity", "confirm"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Lê um CSV UTF-8 e devolve suas colunas e linhas."""
    assert path.is_file(), f"Arquivo ausente: {path}"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _assert_numeric_unique_ids(rows: list[dict[str, str]], filename: str) -> None:
    """Exige IDs inteiros positivos e sem repetição."""
    ids = [row["id"].strip() for row in rows]
    assert all(value.isdigit() and int(value) > 0 for value in ids), (
        f"{filename}: todos os IDs devem ser números inteiros positivos"
    )
    assert len(ids) == len(set(ids)), f"{filename}: existem IDs duplicados"


def test_symptoms_sample_columns_and_content() -> None:
    """Valida estrutura, IDs e correspondência da classe com a cultura."""
    columns, rows = _read_csv(SYMPTOMS_CSV)
    assert columns == SYMPTOMS_COLUMNS
    assert rows, "symptoms_sample.csv está vazio"
    _assert_numeric_unique_ids(rows, "symptoms_sample.csv")

    for row in rows:
        assert all(value and value.strip() for value in row.values()), (
            f"Linha com campo vazio: {row}"
        )
        crop = row["crop"].strip()
        disease_class = row["disease_class"].strip()
        assert crop in CROP_CLASS_PREFIX, f"Cultura não reconhecida: {crop!r}"
        assert disease_class.startswith(CROP_CLASS_PREFIX[crop]), (
            "A classe deve corresponder à cultura no padrão PlantVillage: "
            f"{crop!r} / {disease_class!r}"
        )


def test_question_intents_columns_and_content() -> None:
    """Valida estrutura, IDs únicos e conjunto permitido de intenções."""
    columns, rows = _read_csv(INTENTS_CSV)
    assert columns == INTENTS_COLUMNS
    assert rows, "question_intents.csv está vazio"
    _assert_numeric_unique_ids(rows, "question_intents.csv")

    for row in rows:
        intent = row["intent"].strip()
        question = row["question"].strip()
        assert intent, f"Intenção vazia na linha: {row}"
        assert question, f"Pergunta vazia na linha: {row}"
        assert intent in ALLOWED_INTENTS, f"Intenção não permitida: {intent!r}"
