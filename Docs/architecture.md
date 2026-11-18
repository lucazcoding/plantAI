# Arquitetura — PlantAI

```
Imagem ─► VisionAgent (qualidade: nitidez/brilho; MobileNetV2 INT8 → 1280-d + multiescala 1984-d)
              └─► MLAgent (StandardScaler + LogReg: `models/plantai_head.joblib`)
Texto  ─► split_user_message ─► NLPAgent (TF-IDF + LR/LinearSVC)  e  QuestionIntentClassifier (5 intenções)
                                   │
              MLAgent + NLPAgent ─► CoordinatorAgent ─► DecisionRulesEngine (R01–R06) ─► QAResponse
```

Contrato de dados em `src/contracts.py` (classes, saídas dos agentes, `CLASS_QA_INFO` com prevenção e gravidade).

## Regras de decisão
- R01 rejeita imagem de baixa qualidade; R02 usa apenas visão ou apenas texto quando falta a outra evidência; R03 reforça a confiança quando visão e texto concordam; R04 desambigua por texto quando a visão é incerta; R05 sinaliza divergência; R06 abstém-se quando a confiança é baixa.

## Fluxo imagem + pergunta
`CoordinatorAgent.answer(imagem, mensagem)` separa sintomas e pergunta, diagnostica e compõe a resposta (causa, prevenção, gravidade, aviso de incerteza). Não há mocks: sem cabeça treinada, `ModelNotTrainedError` é lançado.

Diagrama: `results/plots/arquitetura.png`.
