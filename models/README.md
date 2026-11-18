# Modelos (`models/`)

| Arquivo | O que é | Origem / licença |
| --- | --- | --- |
| `mobilenet_v2_1.0_224_quant.tflite` | MobileNetV2 1.0/224 **pré-treinado no ImageNet**, quantizado INT8 (3,6 MB). Usado como extrator de características (backbone congelado). | [google-coral/test_data](https://github.com/google-coral/test_data) — Apache-2.0. Arquitetura: Sandler et al., 2018. |
| `plantai_head.joblib` | **Cabeça de classificação treinada por este projeto** sobre características do MobileNetV2 com imagens reais do PlantVillage (≈100 KB). Gerada por `python -m src.ml.run_all_experiments`. | Produzida pelo projeto. |
| `mobilenetv2_plantvillage.keras` | (opcional, não versionado) fine-tuning completo em Keras, gerado no Colab por `scripts/colab_finetune_mobilenetv2.py`. | — |

## Como regenerar
```bash
bash scripts/download_datasets.sh          # PlantVillage + PlantDoc reais (via git, ~350 MB)
python -m src.ml.run_all_experiments       # ~3 min em CPU: extrai características, treina, avalia e grava a cabeça
```
O `VisionAgent` carrega `plantai_head.joblib` automaticamente. Se o arquivo não existir, a inferência
levanta `ModelNotTrainedError` — o sistema **não inventa probabilidades**.
