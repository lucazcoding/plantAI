"""Script de demonstração ponta a ponta do Sistema Multiagente PlantAI.

Executa:
1. Treinamento e avaliação científica (5-Fold Stratified CV) do módulo Baseline de PLN
   (`TF-IDF + LogisticRegression`) sobre `data/symptoms_sample.csv`.
2. Simulação dos 4 cenários principais no `CoordinatorAgent`:
   - Cenário 1: Concordância Total (Visão + PLN convergem com boost de confiança)
   - Cenário 2: Desempate por PLN (Visão em dúvida moderada no Top-K desempatada pelo texto)
   - Cenário 3: Divergência Crítica (Conflito explícito entre Visão e PLN sem ocultar erro)
   - Cenário 4: Somente Imagem (Fallback automático quando o usuário não envia texto)
3. Demonstração do Estudo de Ablação (Visão Isolada vs. PLN Isolado vs. Sistema Integrado).
"""

from __future__ import annotations

import sys

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.nlp_agent import NLPAgent
from src.agents.vision_agent import VisionAgent
from src.vision.image_processor import ImageProcessor


def main() -> None:
    # Garante saída UTF-8 limpa em terminais Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 78)
    print("  PLANTAI — INICIALIZANDO PIPELINE MULTIAGENTE E TREINAMENTO BASELINE PLN")
    print("=" * 78)

    nlp_agent = NLPAgent(algorithm="logistic_regression", auto_train_from_sample=True)
    vision_agent = VisionAgent()
    coordinator = CoordinatorAgent(vision_agent=vision_agent, nlp_agent=nlp_agent)

    metrics = nlp_agent.evaluation_metrics
    if metrics is not None:
        print(f"Modelo PLN Treinado : {metrics.algorithm}")
        print(f"Amostras no Dataset : {metrics.num_samples} descrições (data/symptoms_sample.csv)")
        print(f"Classes Cobertas    : {len(metrics.class_names)} classes do PlantVillage")
        print(
            f"Métricas (5-Fold CV): "
            f"Accuracy={metrics.accuracy:.2%} | "
            f"Precision={metrics.precision_macro:.2%} | "
            f"Recall={metrics.recall_macro:.2%} | "
            f"F1-Score={metrics.f1_macro:.2%}"
        )
    print()

    sharp_leaf = ImageProcessor.create_synthetic_leaf_image("healthy_sharp")

    # -------------------------------------------------------------------------
    # CENÁRIO 1: CONCORDÂNCIA TOTAL (Visão + PLN)
    # -------------------------------------------------------------------------
    d1 = coordinator.diagnose(
        image_input=sharp_leaf,
        symptom_text=(
            "Folhas mais velhas do tomateiro apresentando manchas escuras arredondadas "
            "com anéis concêntricos tipo alvo e halo amarelado ao redor."
        ),
        mock_vision_probs={
            "Tomato___Early_blight": 0.88,
            "Tomato___Late_blight": 0.08,
            "Tomato___Healthy": 0.04,
        },
    )
    print(coordinator.generate_report(d1, title="CENÁRIO 1: CONCORDÂNCIA TOTAL"))
    print()

    # -------------------------------------------------------------------------
    # CENÁRIO 2: DESEMPATE POR PLN (Visão em dúvida moderada no Top-K)
    # -------------------------------------------------------------------------
    d2 = coordinator.diagnose(
        image_input=sharp_leaf,
        symptom_text=(
            "Requeima severa no tomate: grandes lesões encharcadas de aspecto oleoso "
            "nas bordas das folhas e formação de mofo esbranquiçado na face inferior."
        ),
        mock_vision_probs={
            "Tomato___Early_blight": 0.51,
            "Tomato___Late_blight": 0.44,
            "Tomato___Healthy": 0.05,
        },
    )
    print(coordinator.generate_report(d2, title="CENÁRIO 2: DESEMPATE VISUAL PELO PLN"))
    print()

    # -------------------------------------------------------------------------
    # CENÁRIO 3: DIVERGÊNCIA CRÍTICA (Conflito explícito entre Visão e PLN)
    # -------------------------------------------------------------------------
    d3 = coordinator.diagnose(
        image_input=sharp_leaf,
        symptom_text=(
            "Folhas de milho cobertas por numerosas pústulas salientes marrom-avermelhadas "
            "cor de canela e ferrugem nas duas faces da lâmina foliar."
        ),
        mock_vision_probs={
            "Tomato___Healthy": 0.91,
            "Tomato___Early_blight": 0.06,
            "Tomato___Late_blight": 0.03,
        },
    )
    print(coordinator.generate_report(d3, title="CENÁRIO 3: DIVERGÊNCIA CRÍTICA"))
    print()

    # -------------------------------------------------------------------------
    # CENÁRIO 4: SOMENTE IMAGEM (Fallback sem texto do usuário)
    # -------------------------------------------------------------------------
    d4 = coordinator.diagnose(
        image_input=sharp_leaf,
        symptom_text="",
        mock_vision_probs={
            "Potato___Late_blight": 0.86,
            "Potato___Early_blight": 0.10,
            "Tomato___Late_blight": 0.04,
        },
    )
    print(coordinator.generate_report(d4, title="CENÁRIO 4: MODO SOMENTE IMAGEM"))
    print()

    # -------------------------------------------------------------------------
    # BÔNUS CIENTÍFICO: ESTUDO DE ABLAÇÃO (Visão vs. PLN vs. Integrado)
    # -------------------------------------------------------------------------
    ablation = coordinator.run_ablation_comparison(
        image_input=sharp_leaf,
        symptom_text=(
            "Folhas de batata com manchas encharcadas escuras nas margens e mofo branco "
            "na face inferior, típicas de requeima da batateira."
        ),
        mock_vision_probs={
            "Potato___Early_blight": 0.54,
            "Potato___Late_blight": 0.41,
            "Tomato___Late_blight": 0.05,
        },
    )
    print("=" * 78)
    print("  RESUMO DO ESTUDO DE ABLAÇÃO (CASO COM AMBIGUIDADE VISUAL)")
    print("  Verdadeiro (Ground Truth): Potato___Late_blight")
    print("-" * 78)
    for mode_name, result in ablation.items():
        print(
            f"  Modo: {mode_name:<12} | Status: {result.status.value:<36} | "
            f"Classe Final: {str(result.final_class):<22} | Conf: {result.combined_confidence:.2%}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()

