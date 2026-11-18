"""Ablação do Experimento 4: efeito do reforço por cultura citada no PLN de sintomas.

Compara, com a MESMA validação cruzada 5-fold (textos sempre avaliados fora da dobra):
- ``pure``: TF-IDF + modelo linear, sem regra adicional;
- ``crop_boost``: o NLPAgent como é implantado (TF-IDF + modelo linear + reforço ×1,35/×0,75
  quando exatamente uma cultura é citada no texto).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.nlp.symptom_classifier import CROP_BOOST_MATCH, CROP_BOOST_OTHER, SymptomClassifier

ROOT = Path(__file__).resolve().parents[2]


def run(seed: int = 42, n_splits: int = 5) -> dict[str, Any]:
    df = pd.read_csv(ROOT / "data" / "symptoms_sample.csv")
    texts = df["symptom_description"].astype(str).tolist()
    labels = df["disease_class"].astype(str).tolist()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: dict[str, Any] = {
        "n": len(texts),
        "cv": f"StratifiedKFold({n_splits}, shuffle=True, random_state={seed})",
        "crop_boost_factors": {"match": CROP_BOOST_MATCH, "other": CROP_BOOST_OTHER},
    }
    for algo, tag in (("logistic_regression", "logreg"), ("svm", "svm")):
        for boost in (False, True):
            pred = [""] * len(texts)
            for tr, te in skf.split(texts, labels):
                clf = SymptomClassifier(algorithm=algo, crop_boost=boost).train(  # type: ignore[arg-type]
                    [texts[i] for i in tr], [labels[i] for i in tr]
                )
                for i in te:
                    pred[i] = clf.predict(texts[i])["predicted_class"]
            key = f"{tag}_{'crop_boost' if boost else 'pure'}"
            out[key] = {
                "accuracy": round(float(accuracy_score(labels, pred)), 4),
                "f1_macro": round(float(f1_score(labels, pred, average="macro")), 4),
            }
    return out


def main() -> None:
    res = run()
    path = ROOT / "results" / "metrics" / "exp4_nlp_crop_boost_ablation.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
