"""Fine-tuning completo do MobileNetV2 (Keras/TensorFlow) — para rodar no Google Colab (GPU opcional).

ATENÇÃO: este script NÃO foi executado no ambiente em que o repositório foi auditado/corrigido (sem TensorFlow
e sem GPU). Os resultados reportados no README e no artigo vêm de `python -m src.ml.run_all_experiments`
(MobileNetV2 congelado + cabeça treinada). Rode este script no Colab se quiser comparar com o fine-tuning
completo; ele usa EXATAMENTE a mesma divisão treino/val/teste (results/metrics/split_manifest.json).

Colab:
    !git clone <URL_DO_SEU_REPOSITORIO> plantAI && cd plantAI
    !pip install -q "tensorflow>=2.15"
    !bash scripts/download_datasets.sh
    !python scripts/colab_finetune_mobilenetv2.py --epochs-head 5 --epochs-finetune 5
Saídas: models/mobilenetv2_plantvillage.keras e results/metrics/colab_finetune_mobilenetv2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Corn_(maize)___Common_rust_",
]
IMG = 224


def make_ds(paths: list[str], train: bool, batch: int) -> tf.data.Dataset:
    labels = [CLASSES.index(Path(p).parent.name) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices(([str(ROOT / p) for p in paths], labels))

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        return tf.cast(img, tf.float32), tf.one_hot(y, len(CLASSES))

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"), tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.2), tf.keras.layers.RandomContrast(0.2), tf.keras.layers.RandomBrightness(0.2),
        ])
        ds = ds.shuffle(2048, seed=42).map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)

    manifest = json.loads((ROOT / "results" / "metrics" / "split_manifest.json").read_text(encoding="utf-8"))
    train_ds, val_ds, test_ds = (make_ds(manifest[k], k == "train", args.batch) for k in ("train", "val", "test"))

    from src.vision.classifier import MobileNetV2Classifier  # reutiliza a arquitetura do projeto

    model = MobileNetV2Classifier.build_mobilenetv2_architecture(len(CLASSES), freeze_backbone=True)
    # O modelo do projeto usa Rescaling(2, -1): entrada esperada em [0, 1] -> convertemos de [0, 255].
    inputs = tf.keras.Input((IMG, IMG, 3))
    wrapped = tf.keras.Model(inputs, model(tf.keras.layers.Rescaling(1 / 255.0)(inputs)))
    wrapped.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", ["accuracy"])
    h1 = wrapped.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head)

    base = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-30]:  # descongela só os últimos 30 blocos/camadas
        layer.trainable = False
    wrapped.compile(tf.keras.optimizers.Adam(1e-5), "categorical_crossentropy", ["accuracy"])
    h2 = wrapped.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune)

    y_true = np.concatenate([np.argmax(y, 1) for _, y in test_ds])
    y_pred = np.argmax(wrapped.predict(test_ds), 1)
    result = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)), "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True),
        "history_head": h1.history, "history_finetune": h2.history, "tensorflow": tf.__version__,
    }
    (ROOT / "results" / "metrics" / "colab_finetune_mobilenetv2.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (ROOT / "models").mkdir(exist_ok=True)
    wrapped.save(ROOT / "models" / "mobilenetv2_plantvillage.keras")
    print(f"Teste: acc={result['test_accuracy']:.4f} F1={result['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
