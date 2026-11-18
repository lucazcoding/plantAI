# Tabelas de resultados (geradas automaticamente)

Geradas por `scripts/make_report_assets.py` a partir de `results/metrics/*.json`.

| Modelo | Acurácia (%) | F1-macro (%) | IC95% acurácia |
|---|--:|--:|---|
| Classe majoritária | 24,8 | 6,6 | — |
| 6 estatísticas RGB + LogReg | 60,4 | 62,2 | [57,6; 63,4] |
| 30 descritores clássicos + MLP | 96,7 | 96,6 | [95,6; 97,7] |
| MobileNetV2 (1280-d final) + LogReg | 95,1 | 95,0 | — |
| MobileNetV2 multiescala + cabeça (implantado) | 97,6 | 97,5 | [96,6; 98,5] |

| Modelo PLN | N | Acurácia (%) | F1-macro (%) |
|---|--:|--:|--:|
| TF-IDF + LogReg (puro, sem regra de cultura) | 180 | 80,6 | 80,6 |
| TF-IDF + LinearSVC calibrado (puro, sem regra de cultura) | 180 | 81,1 | 81,2 |
| NLPAgent: TF-IDF + LogReg + reforço de cultura (implantado) | 180 | 82,8 | 82,8 |
| NLPAgent: TF-IDF + LinearSVC + reforço de cultura | 180 | 83,9 | 83,9 |
| Intenção da pergunta | 150 | 84,7 | 84,7 |

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
