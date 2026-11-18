"""Demonstração ponta a ponta do PlantAI com imagens REAIS e modelo REAL (sem probabilidades simuladas).

    python demo_pipeline.py                              # roda 5 cenários com as imagens de results/exemplos/
    python demo_pipeline.py --image minha_folha.jpg --message "Manchas em anel nas folhas de baixo. Como tratar?"

Fluxo (Seção 4.5 do PDF): imagem -> visão (MobileNetV2 + cabeça treinada) -> mensagem do usuário ->
PLN (sintomas + intenção da pergunta) -> regras de decisão -> resposta contextualizada.
O relatório completo é salvo em results/exemplos/demo_output.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agents.coordinator_agent import CoordinatorAgent

EXEMPLOS = Path(__file__).resolve().parent / "results" / "exemplos"

SCENARIOS = [
    ("1. Concordância (imagem real + relato correto + pergunta de tratamento)", "plantvillage_Tomato_Late_blight.jpg",
     "As folhas do tomateiro estão com manchas grandes encharcadas e mofo branco embaixo. Como tratar?"),
    ("2. Concordância em outra cultura (pergunta sobre gravidade)", "plantvillage_Corn_maize_Common_rust.jpg",
     "Pústulas marrom-avermelhadas salientes que soltam pó de ferrugem nas folhas do milho. Isso é grave?"),
    ("3. Divergência: relato (milho) contradiz a foto (tomate saudável)", "plantvillage_Tomato_healthy.jpg",
     "Folhas de milho cobertas por pústulas marrom-avermelhadas cor de ferrugem. Qual doença é essa?"),
    ("4. Somente imagem (sem relato), pergunta de prevenção", "plantvillage_Potato_Early_blight.jpg",
     "Como evitar que isso volte?"),
    ("5. Foto de campo (PlantDoc, fundo não controlado) + relato + pergunta de confirmação", "plantdoc_Tomato_Early_blight.jpg",
     "Folhas de baixo com manchas marrons em anéis tipo alvo e amarelando. Será que é pinta-preta?"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, help="Caminho de uma foto de folha.")
    ap.add_argument("--message", type=str, default="", help="Relato de sintomas e/ou pergunta.")
    args = ap.parse_args()

    coordinator = CoordinatorAgent()
    m = coordinator.nlp_agent.evaluation_metrics
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 78)
    emit("  PLANTAI — DEMONSTRAÇÃO COM MODELOS TREINADOS EM DADOS REAIS")
    emit("=" * 78)
    emit(f"Visão : {coordinator.ml_agent.classifier.head_description}")
    if m is not None:
        emit(f"PLN   : {m.algorithm} | CV 5-fold em {m.num_samples} textos: acc={m.accuracy:.2%}, F1={m.f1_macro:.2%}")
    emit()

    cases = [("Entrada do usuário", args.image, args.message)] if args.image else [(t, EXEMPLOS / img, msg) for t, img, msg in SCENARIOS]
    for title, img, msg in cases:
        resp = coordinator.answer(img, msg)
        emit("#" * 78)
        emit(f"# {title}")
        emit(f"# Imagem  : {Path(img).name}")
        emit(f"# Mensagem: {msg}")
        emit(f"# Intenção da pergunta: {resp.intent} (conf={resp.intent_confidence:.0%}) | sintomas: '{resp.symptom_text}'")
        emit("#" * 78)
        emit(coordinator.generate_report(resp.decision))
        emit(f"\nRESPOSTA AO USUÁRIO:\n  {resp.answer}\n")

    if not args.image:
        out = EXEMPLOS / "demo_output.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        coordinator.export_experimental_history(EXEMPLOS / "demo_history.json")
        print(f"[salvo] {out}")


if __name__ == "__main__":
    main()
