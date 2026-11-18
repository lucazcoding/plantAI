# Metodologia — PlantAI

1. **Dados:** PlantVillage (6 classes) e PlantDoc (externo) obtidos por `scripts/download_datasets.sh`. Limpeza (arquivos corrompidos, duplicatas MD5/dHash) e divisão ≈71/14/14 (alvo nominal 70/15/15) agrupada por folha.
2. **Visão/ML:** MobileNetV2 1.0/224 INT8 congelado (ImageNet) como extrator; cabeça LogReg/MLP escolhida pelo F1 de validação; teste usado uma única vez. Baselines: maioria, RGB, 30 descritores clássicos.
3. **Aumento de dados:** adotado após baixo desempenho no PlantDoc, com regra de adoção fixada a priori (ajuste pós-observação, declarado).
4. **PLN:** 180 textos (30 por classe) e 150 perguntas escritos pela equipe; TF-IDF + LR/LinearSVC; avaliação 5-fold out-of-fold.
5. **Integração:** regras R01–R06; métricas estritas (abstenção conta como erro), cobertura e precisão-quando-responde; IC95% por bootstrap; varredura de relatos errados.
6. **Reprodutibilidade:** sementes fixas, `results/metrics/environment.json`, testes `pytest`, notebooks 00–07 executados.
7. **Limitações:** ver README §8.
