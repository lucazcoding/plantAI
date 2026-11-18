"""Testes de ML: limpeza/divisão agrupada dos dados e reprodução das métricas gravadas em results/."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.ml import dataset as ds
from tests.helpers import ROOT


def _noise_image(path, seed: int) -> None:
    arr = np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_cleaning_removes_corrupted_and_exact_duplicates(tmp_path) -> None:
    folder = tmp_path / ds.CLASS_NAMES[0]
    folder.mkdir()
    _noise_image(folder / "a.png", 1)
    (folder / "a_copy.png").write_bytes((folder / "a.png").read_bytes())  # duplicata exata
    _noise_image(folder / "b.png", 2)
    (folder / "broken.png").write_bytes(b"not an image")
    records = ds.list_records(tmp_path, use_leaf_groups=False)
    kept, report = ds.clean_records(records)
    assert report.total_listed == 4 and report.corrupted_removed == 1 and report.exact_duplicates_removed == 1
    assert len(kept) == 2


def test_split_is_stratified_grouped_and_leak_free() -> None:
    rng = np.random.default_rng(0)
    recs = []
    for cls in ds.CLASS_NAMES:
        for g in range(60):
            for k in range(3):  # 3 fotos por folha
                recs.append(ds.ImageRecord(path=ds.ROOT / f"{cls}_{g}_{k}.jpg", label=cls, group=f"{cls}:{g}"))
    rng.shuffle(recs)
    split = ds.split_70_15_15(recs, seed=42)
    assert sum(len(v) for v in split.values()) == len(recs)
    assert all(v == 0 for v in ds.verify_no_group_leakage(recs, split).values())
    assert 0.65 < len(split["train"]) / len(recs) < 0.78


def test_recorded_metrics_are_consistent_with_real_data() -> None:
    metrics = ROOT / "results" / "metrics"
    split = json.loads((metrics / "dataset_split_70_15_15.json").read_text(encoding="utf-8"))
    assert split["group_leakage_between_partitions"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert split["train_count"] + split["val_count"] + split["test_count"] == split["cleaning"]["valid_unique_images"]
    exp2 = json.loads((metrics / "exp2_transfer_learning.json").read_text(encoding="utf-8"))
    exp1 = json.loads((metrics / "exp1_baseline_vision.json").read_text(encoding="utf-8"))
    assert exp2["accuracy"] > exp1["accuracy"] + 0.2  # transfer learning supera o baseline por ampla margem


@pytest.mark.skipif(not ds.dataset_available(), reason="rode scripts/download_datasets.sh para baixar o PlantVillage")
def test_reproduces_test_accuracy_from_saved_head() -> None:
    """Reavalia o modelo salvo no conjunto de teste real e confere com o JSON gravado."""
    from sklearn.metrics import accuracy_score

    from src.vision.classifier import MobileNetV2Classifier

    records, _ = ds.clean_records(ds.list_records(ds.PLANTVILLAGE_DIR))
    split = ds.split_70_15_15(records, seed=42)
    feats = ds.extract_features(records, "plantvillage_features")
    clf = MobileNetV2Classifier()
    x = feats["multiscale"] if clf.feature_mode == "multiscale" else feats["embeddings"]
    acc = accuracy_score(feats["labels"][split["test"]], clf.head.predict(x[split["test"]]))
    recorded = json.loads((ROOT / "results" / "metrics" / "exp2_transfer_learning.json").read_text(encoding="utf-8"))["accuracy"]
    assert abs(acc - recorded) < 1e-3
