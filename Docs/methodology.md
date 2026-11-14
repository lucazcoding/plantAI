# Metodologia Científica e Experimental — PlantAI

**Instituição:** AFYA | UNIMA Maceió  
**Disciplina:** Inteligência Artificial Aplicada — Prof. Ícaro Ferreira  

---

## 1. Preparação e Divisão Estratificada dos Dados (70% / 15% / 15%)

Seguindo as Seções 3.3 e 3.4 do planejamento científico:
1. **Verificação de Integridade:** Imagens corrompidas ou ilegíveis são detectadas e descartadas antes da extração de atributos.
2. **Prevenção de Vazamento de Dados (*Data Leakage*):** Cada imagem recebe uma assinatura perceptual SHA-256 sobre grade reduzida (`8x8`) para remoção de arquivos duplicados ou quase-duplicados antes da partição.
3. **Divisão Estratificada:**
   - **Treinamento (70%):** ajuste dos pesos dos classificadores.
   - **Validação (15%):** seleção de hiperparâmetros e monitoramento por época.
   - **Teste (15%):** mantido estritamente isolado para cálculo das métricas finais.

---

## 2. Modelos de Visão Computacional e Machine Learning

- **Modelo Baseline (Experimento 1):** Regressão Logística regularizada operando apenas sobre estatísticas globais básicas de cor, estabelecendo a linha de base inferior.
- **Arquitetura MobileNetV2 e Cabeça de Transfer Learning (Experimento 2):**
  - Entrada padronizada `224x224x3`, camada `GlobalAveragePooling2D`, `Dropout(0.2)` e camada densa `Softmax` com função de perda `Categorical Crossentropy` e otimizador `Adam (lr=0.001)`.
  - Extração conjunta de 30 descritores agronômicos e espectrais alimentando a cabeça neural supervisionada (`MLPClassifier (64, 32)`).

---

## 3. Processamento de Linguagem Natural (PLN)

- **Dataset (`data/symptoms_sample.csv`):** 60 descrições de sintomas agronômicos em português brasileiro distribuídas equilibradamente entre 6 classes do PlantVillage.
- **Pré-processamento (`TextProcessor`):** remoção de acentos, normalização morfológica de culturas (*tomateiro → tomate*, *batateira → batata*, *milharal → milho*), preservação de negações e extração de léxico fitossanitário.
- **Classificadores Avaliados (5-Fold Stratified Cross-Validation):**
  1. `TF-IDF (1,2-gramas) + LogisticRegression (C=12.0, class_weight='balanced')`
  2. `TF-IDF (1,2-gramas) + CalibratedClassifierCV(LinearSVC)`

---

## 4. Heurísticas e Regras de Decisão (`DecisionRulesEngine`)

| Código da Regra | Condição de Acionamento | Status Emitido |
|---|---|---|
| `R01_IMAGE_QUALITY_REJECTION` | Imagem corrompida, escura (`brilho < 25`) ou borrada (`nitidez < 18`) | `QUALIDADE_IMAGEM_INSUFICIENTE` |
| `R02_VISION_ONLY_HIGH_CONFIDENCE` | Sem texto do usuário e confiança visual `>= 0.75` | `DIAGNOSTICO_SOMENTE_VISUAL` |
| `R02_VISION_ONLY_LOW_CONFIDENCE` | Sem texto do usuário e confiança visual `< 0.75` | `INCONCLUSIVO_BAIXA_CONFIANCA` |
| `R03_EXACT_CONCORDANCE_BOOST` | `VisionAgent` e `NLPAgent` convergem na mesma classe Top-1 | `CONFIRMADO_POR_MULTIPLAS_EVIDENCIAS` |
| `R04_NLP_DISAMBIGUATES_MODERATE_VISION` | Visão com dúvida moderada (`0.45` a `0.75`) no Top-K e PLN confiante (`>= 0.70`) em uma das classes candidatas | `DESAMBIGUADO_POR_EVIDENCIA_TEXTUAL` |
| `R05_EXPLICIT_AGENT_DIVERGENCE` | Visão e PLN apontam classes conflitantes sem suporte cruzado | `DIVERGENCIA_DETECTADA` |
| `R06_GENERAL_UNCERTAINTY_LOW_CONFIDENCE` | Ambos os agentes com confiança `< 0.45` | `INCONCLUSIVO_BAIXA_CONFIANCA` |

