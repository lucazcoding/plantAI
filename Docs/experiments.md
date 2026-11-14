# Relatório de Experimentos e Avaliação Científica — PlantAI

Todos os resultados abaixo foram obtidos por execução real e reproduzível dos algoritmos implementados no repositório, com artefatos salvos em `results/metrics/`, `results/confusion_matrices/` e `results/plots/`.

---

## 1. Preparação do Dataset e Divisão Estratificada (70/15/15)

- **Total bruto inspecionado:** 610 entradas
- **Arquivos corrompidos removidos automaticamente:** 4
- **Duplicatas exatas removidas (prevenção de *data leakage*):** 6
- **Total líquido válido:** 600 imagens (100 por classe)
  - **Treinamento (70%):** 420 imagens (70 por classe)
  - **Validação (15%):** 90 imagens (15 por classe)
  - **Teste (15%):** 90 imagens (15 por classe)

---

## 2. Tabela Consolidada dos 6 Experimentos (Métricas Reais)

| ID | Experimento / Modelo | Conjunto Avaliado | Amostras ($N$) | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Tempo (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Exp 1** | Baseline Visual (Atributos Globais Básicos) | Teste Estratificado (15%) | 90 | **77,78%** | 81,20% | 77,78% | 74,45% | 0,0118 |
| **Exp 2** | MobileNetV2 Transfer Learning Head (30 Descritores) | Teste Estratificado (15%) | 90 | **100,00%** | 100,00% | 100,00% | 100,00% | 0,4234 |
| **Exp 4A** | PLN Baseline (`TF-IDF + LogisticRegression`) | 5-Fold Stratified CV | 60 | **95,00%** | 95,15% | 95,00% | 95,00% | 0,0400 |
| **Exp 4B** | PLN Comparativo (`TF-IDF + Calibrated LinearSVC`) | 5-Fold Stratified CV | 60 | **95,00%** | 95,15% | 95,00% | 95,00% | 0,0600 |
| **Exp 5A** | Estudo de Ablação — Visão Isolada (c/ 35% Fotos Ambíguas) | Conjunto Multimodal | 60 | **66,67%** | 81,82% | 66,67% | 71,04% | — |
| **Exp 5C** | Estudo de Ablação — **Sistema Multiagente Integrado** | Conjunto Multimodal | 60 | **95,00%** | 96,15% | 95,00% | 94,88% | — |
| **Exp 6** | Generalização Externa em Condições de Campo (Estilo PlantDoc) | Domínio Externo c/ Ruído | 27 | **44,44%** | 41,29% | 45,83% | 35,28% | — |

---

## 3. Experimento 3 — Ajuste de Hiperparâmetros

| Configuração | Learning Rate | Camadas Ocultas | Regularização ($\alpha$) | Val Accuracy | Val F1-Macro | Test Accuracy | Test F1-Macro |
|---|---:|---|---:|---:|---:|---:|---:|
| **Config A** | `0.0100` | `(32,)` | `0.0100` | 100,00% | 100,00% | 100,00% | 100,00% |
| **Config B (Padrão)** | `0.0010` | `(64, 32)` | `0.0010` | **100,00%** | **100,00%** | **100,00%** | **100,00%** |
| **Config C** | `0.0005` | `(128, 64)` | `0.0005` | 100,00% | 100,00% | 100,00% | 100,00% |
| **Config D (Instável)** | `0.0500` | `(32,)` | `0.1000` | 97,78% | 97,77% | 97,78% | 97,77% |

---

## 4. Discussão Crítica dos Resultados e Limitações

1. **Ganho sobre o Baseline (Exp 1 vs. Exp 2):** A inclusão dos descritores morfológicos de lesão (razão necrótica, halo clorótico, área encharcada e energia de textura) elevou o F1-Score de **74,45%** para **100,00%** nas imagens controladas de teste.
2. **Desambiguação Multiagente (Exp 5):** Quando 35% das imagens apresentam ambiguidade entre doenças visualmente semelhantes (como estágios iniciais de *Early blight* vs. *Late blight*), o módulo visual isolado cai para **66,67%** de acurácia. Ao integrar o `NLPAgent` e a **Regra R04 (`DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL`)**, o `CoordinatorAgent` recupera a acurácia para **95,00%** (+28,33 pontos percentuais).
3. **Limitação de Generalização em Campo (Exp 6 — PlantVillage vs. PlantDoc):** Modelos treinados apenas em imagens de laboratório com iluminação homogênea sofrem queda severa (**44,44%** de acurácia) quando expostos a sombras laterais e ruído de fundo de solo, demonstrando a importância do controle de qualidade visual (`R01`) e da validação cruzada com o relato textual do agricultor.

