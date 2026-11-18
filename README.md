# PlantAI — Sistema Multiagente para Identificação e Análise de Doenças em Plantas

**Instituição:** AFYA | UNIMA Maceió  
**Disciplina:** Inteligência Artificial  
**Professor:** Ícaro Ferreira  
**Ambiente:** Python 3.10+ / Google Colab / GitHub / Overleaf (IEEE)  
**Artigo (IEEE, ≤ 6 págs.):** `artigo/main.tex` (compilar no Overleaf com IEEEtran) — prévia: `artigo/PlantAI_artigo_preview.pdf` — **Overleaf: [abrir o artigo no Overleaf](https://www.overleaf.com/project/6afdb72978af010f8bd1f9fd/share#f3c01366aedc227848d8f0fbcec891abc989776660197f72)**

---

## 1. Integrantes e Responsabilidades

| Integrante | Cargo | Responsabilidades no Projeto |
|---|---|---|
| **Lucas Gomes** | **Tech Lead** | Liderança técnica, governança do repositório, documentação dos notebooks e artigo (IEEE) |
| **Viny Carnaúba** | **Desenvolvedor** | Arquitetura multiagente (`contracts.py`), `CoordinatorAgent`, `DecisionRulesEngine`, relatórios e testes integrados |
| **Fabiano Nasiazeno** | **Desenvolvedor** | Preparação e divisão do PlantVillage, `VisionAgent`, `MLAgent` / MobileNetV2 |
| **José Nogueira** | **Desenvolvedor** | Corpus de sintomas, pré-processamento textual, `NLPAgent` (TF-IDF + LR/SVM) |
| **Marco Wichmann** | **Apoio (Testes e Documentação)** | Testes automatizados (`tests/`), curadoria de exemplos (`data/`) e revisão de documentação (`Docs/`) |

---

## 2. Problema e Objetivos

Doenças foliares em tomate, batata e milho causam perdas agrícolas e exigem diagnóstico precoce. O PlantAI recebe **uma foto da folha e, opcionalmente, uma mensagem em português** (sintomas e/ou pergunta) e devolve um diagnóstico com explicação, usando:

1. **Visão computacional**: controle de qualidade (nitidez, brilho) e extração de características com MobileNetV2 pré-treinado (ImageNet).
2. **Machine Learning**: cabeça supervisionada (LogReg) sobre características multiescala, comparada a baselines.
3. **PLN**: TF-IDF + LR/LinearSVC para sintomas e classificador de intenção da pergunta (5 intenções).
4. **Decisão multiagente**: 6 regras explícitas (R01–R06) combinam evidências, desambiguam, **alertam divergência** e abstêm-se quando incerto.

Classes (6): Tomate (requeima, pinta-preta, saudável), Batata (requeima, pinta-preta), Milho (ferrugem comum).

## 3. Tecnologias

Python, NumPy, scikit-learn, Pillow, matplotlib, `ai-edge-litert` (MobileNetV2 1.0/224 INT8 TFLite), pytest, nbclient (notebooks), LaTeX IEEEtran. TensorFlow é opcional (apenas para o script de fine-tuning no Colab).

## 4. Arquitetura

`Imagem → VisionAgent (qualidade + embeddings) → MLAgent (cabeça)` e `Texto → NLPAgent (sintomas + intenção)` → `CoordinatorAgent (DecisionRulesEngine R01–R06)` → resposta. Diagrama: `results/plots/arquitetura.png`; detalhes em `Docs/architecture.md`.

```
plantAI/
├── src/ (contracts, agents/, vision/, ml/, nlp/, decision/)
├── data/ (symptoms_sample.csv, question_intents.csv; imagens baixadas por script)
├── models/ (mobilenet_v2 tflite, plantai_head.joblib)
├── notebooks/ (00–07 executados)   ├── tests/   ├── scripts/
├── results/ (metrics/, plots/, exemplos/)   ├── Docs/   └── artigo/
```

> **Nota sobre nomes de pastas:** o enunciado sugere `src/visao_computacional`, `pln`, `machine_learning` e `decisao`; neste repositório os equivalentes são `src/vision`, `src/nlp`, `src/ml` e `src/decision` (nomes em inglês, mesma responsabilidade).

## 5. Instalação e Execução

```bash
pip install -r requirements.txt
bash scripts/download_datasets.sh          # PlantVillage (6 classes) e PlantDoc (git sparse clone)
pytest -q                                   # testes
python demo_pipeline.py                     # demo com imagens reais (ou --image F.jpg --message "texto")
python -m src.ml.nlp_ablation                # só PLN (rápido, sem imagens): TF-IDF puro vs. com regra de cultura
python -m src.ml.run_all_experiments        # Experimentos 1–6 -> results/metrics, plots, models/
python scripts/build_notebooks.py           # executa notebooks 00–07
python scripts/make_report_assets.py        # números/tabelas/figuras do artigo
```

## 6. Modelos

- **Backbone:** MobileNetV2 1.0/224 INT8 (ImageNet, Apache-2.0), **congelado**; características finais (1280-d) + médias/desvios de blocos intermediários (multiescala, 1984-d).
- **Cabeça implantada:** StandardScaler + LogReg (C=0,1), com aumento de dados (2 cópias), selecionada na validação (regra definida a priori).
- **PLN:** TF-IDF + LogReg / LinearSVC calibrado; intenção da pergunta com TF-IDF palavra+caractere.
- **Fine-tuning completo:** `scripts/colab_finetune_mobilenetv2.py` — **não executado** neste ambiente.

## 7. Resultados (reais, gerados em `results/metrics/*.json`)

Divisão ≈71/14/14 (5484/1097/1097 de 7678 imagens; alvo nominal 70/15/15, ajustado pelo agrupamento por folha), estratificada **por classe e agrupada por folha** (sem vazamento), com deduplicação MD5+dHash. PlantDoc = imagens de campo (domínio externo).


| Modelo | Acurácia (%) | F1-macro (%) | IC95% acurácia |
|---|--:|--:|---|
| Classe majoritária | 24,8 | 6,6 | — |
| 6 estatísticas RGB + LogReg | 60,4 | 62,2 | [57,6; 63,4] |
| 30 descritores clássicos + MLP | 96,7 | 96,6 | [95,6; 97,7] |
| MobileNetV2 (1280-d final) + LogReg | 95,1 | 95,0 | — |
| MobileNetV2 multiescala + cabeça (implantado) | 97,6 | 97,5 | [96,6; 98,5] |

| Modelo PLN | N | Acurácia (%) | F1-macro (%) |
|---|--:|--:|--:|
| TF-IDF + LogReg (**puro**, sem regra de cultura) | 180 | 80,6 | 80,6 |
| TF-IDF + LinearSVC calibrado (**puro**) | 180 | 81,1 | 81,2 |
| NLPAgent: TF-IDF + LogReg + **reforço de cultura** (implantado) | 180 | 82,8 | 82,8 |
| NLPAgent: TF-IDF + LinearSVC + **reforço de cultura** | 180 | 83,9 | 83,9 |
| Intenção da pergunta | 150 | 84,7 | 84,7 |

> **Atenção (PLN):** os 82,8% / 83,9% incluem uma regra auditável no `NLPAgent.predict`: quando o texto cita exatamente uma cultura, as probabilidades das classes dessa cultura são multiplicadas por 1,35 e as demais por 0,75 (renormalizadas). O pipeline TF-IDF + modelo linear **sem** essa regra (`SymptomClassifier(crop_boost=False)`, ou `SymptomClassifier.evaluate_cross_validation`) dá 80,6% / 81,1%. Ambos usam o mesmo CV 5-fold e são reproduzíveis com `python -m src.ml.nlp_ablation` (saída em `results/metrics/exp4_nlp_crop_boost_ablation.json`).

| Conjunto | Modo | Acurácia estrita (%) | Cobertura (%) | Precisão se respondeu (%) | IC95% acurácia |
|---|---|--:|--:|--:|---|
| PlantVillage teste (N=1097) | Visão isolada | 95,3 | 96,4 | 98,8 | [93,9; 96,4] |
| PlantVillage teste (N=1097) | PLN isolado | 76,3 | 87,6 | 87,1 | [73,8; 78,8] |
| PlantVillage teste (N=1097) | Integrado (visão+PLN+regras) | 87,7 | 88,3 | 99,3 | [85,7; 89,6] |
| PlantDoc campo (N=586) | Visão isolada | 39,9 | 83,8 | 47,7 | [36,0; 43,9] |
| PlantDoc campo (N=586) | PLN isolado | 77,6 | 89,8 | 86,5 | [74,4; 81,1] |
| PlantDoc campo (N=586) | Integrado (visão+PLN+regras) | 42,5 | 48,5 | 87,7 | [38,2; 46,4] |

Sensibilidade ao erro do usuário (fração de relatos textuais errados → acurácia estrita / precisão do integrado):

| Conjunto | p=0,1 | p=0,2 | p=0,3 | p=0,5 |
|---|---|---|---|---|
| plantvillage_test | 79,0 / 99,1 | 69,8 / 99,0 | 62,9 / 99,0 | 45,9 / 98,8 |
| plantdoc_field | 38,7 / 85,0 | 35,3 / 78,1 | 32,2 / 80,1 | 23,2 / 71,6 |

| Cenário | Acurácia (%) |
|---|--:|
| PlantVillage teste | 97,6 |
| PlantDoc inteiro, sem adaptação | 43,9 [39,8; 48,0] |
| PlantDoc 50% reservado, sem adaptação | 45,4 |
| PlantDoc 50% reservado, com adaptação | 56,7 [51,2; 62,1] |

Conjuntos de características (cabeça LogReg C=0,1):

| Conjunto | Dim | F1 val | Acc teste | Acc PlantDoc |
|---|--:|--:|--:|--:|
| final | 1280 | 94,9 | 95,1 | 41,5 |
| multiscale | 1984 | 97,2 | 97,7 | 42,1 |
| multiscale+handcrafted | 2014 | 97,2 | 97,9 | 41,6 |

Hiperparâmetros da cabeça (seleção pela validação):

| Configuração | F1 val | F1 teste | Acc PlantDoc |
|---|--:|--:|--:|
| LogReg C=0.0001 | 92,9 | 94,0 | 39,8 |
| LogReg C=0.001 | 96,2 | 96,8 | 41,5 |
| LogReg C=0.01 | 97,0 | 97,4 | 41,6 |
| LogReg C=0.1 ✔ (melhor na validação) | 97,2 | 97,7 | 42,1 |
| LogReg C=1 | 97,0 | 97,7 | 37,2 |
| LogReg C=10 | 97,0 | 97,9 | 35,0 |
| MLP (256) lr=1e-3 | 96,7 | 97,1 | 40,4 |
| MLP (512,128) lr=1e-3 | 95,9 | 97,4 | 38,7 |
| MLP (256) lr=5e-2 (instável) | 95,4 | 96,9 | 40,1 |
| LogReg C=0.1 + aumento de dados (2 cópias) ✔ (implantada) | 97,0 | 97,5 | 43,9 |

Divisão: train=5484, val=1097, test=1097; folhas (grupos)=3956; imagens com folha mapeada=4918/7678.


## 8. Limitações (leia antes de citar os números)

- **Viés do PlantVillage:** descritores clássicos simples chegam a 96,7%, indicando que fundo/condições de captura bastam para classificar; alta acurácia no PlantVillage **não** implica desempenho em campo.
- **Domínio externo:** em PlantDoc a visão acerta apenas ~40–44% e as probabilidades **não são calibradas** fora do domínio.
- **PLN: regra de cultura e corpus pequeno.** Os números de PLN do sistema (82,8/83,9%) incluem o reforço por cultura; o TF-IDF puro dá 80,6/81,1%.
- **PLN otimista:** os textos são redigidos pela equipe (180) e amostrados da classe verdadeira; não há usuários reais. A sensibilidade a relatos errados está na tabela acima.
- **Fusão troca cobertura por precisão:** o integrado responde menos, mas erra menos quando responde.
- **Mapeamento de folhas** cobre 4918 de 7678 imagens; as demais ficam em grupos próprios.
- **Aumento de dados** foi adotado após observar o desempenho baixo no PlantDoc (regra de adoção fixada a priori, mas é um ajuste pós-observação).
- **Fine-tuning** não foi executado.
- Não é ferramenta de diagnóstico agronômico profissional.

## 9. Referências

Hughes & Salathé (2015) PlantVillage; Singh et al. (2020) PlantDoc; Sandler et al. (2018) MobileNetV2; Noyan (2022) viés de fundo no PlantVillage. Lista completa verificada em `artigo/main.tex`.

## 10. Licença

Ver `LICENSE`. Pesos MobileNetV2: Apache-2.0 (google-coral/test_data).
