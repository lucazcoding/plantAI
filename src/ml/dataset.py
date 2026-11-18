"""Carregamento, limpeza e divisão dos datasets REAIS (PlantVillage e PlantDoc).

Etapas (Seções 3.3/3.4 da documentação e "Dados e planejamento" do PDF):
1. Listagem das imagens por classe (`data/plantvillage/<classe>/*.jpg`).
2. Verificação de integridade (arquivos que o Pillow não consegue decodificar são descartados).
3. Remoção de duplicatas exatas (MD5 do arquivo) e quase-duplicatas (dHash 8x8 idêntico).
4. Divisão 70/15/15 **estratificada por classe e agrupada por folha**: o PlantVillage fotografa a
   mesma folha em várias orientações; sem agrupar, fotos da mesma folha caem em treino e teste
   (vazamento de dados). O mapa folha→imagem vem de `leaf-map.json` (Mohanty et al., 2016).
5. Extração (e cache) dos descritores clássicos (30) e dos embeddings MobileNetV2 (1280).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.contracts import PLANTVILLAGE_CLASS_INFO

ROOT = Path(__file__).resolve().parents[2]
PLANTVILLAGE_DIR = ROOT / "data" / "plantvillage"
PLANTDOC_DIR = ROOT / "data" / "plantdoc"
CACHE_DIR = ROOT / "results" / "cache"
CLASS_NAMES: list[str] = list(PLANTVILLAGE_CLASS_INFO.keys())
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class ImageRecord:
    path: Path
    label: str
    group: str


@dataclass(slots=True)
class CleaningReport:
    """Relatório auditável da limpeza do dataset."""

    total_listed: int = 0
    corrupted_removed: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    valid_unique_images: int = 0
    per_class_after_cleaning: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_listed": self.total_listed,
            "corrupted_removed": self.corrupted_removed,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "valid_unique_images": self.valid_unique_images,
            "per_class_after_cleaning": self.per_class_after_cleaning,
        }


def dataset_available(root: Path = PLANTVILLAGE_DIR) -> bool:
    return root.exists() and any((root / c).is_dir() for c in CLASS_NAMES)


def _leaf_key(filename: str) -> str:
    """Chave do `leaf-map.json`: parte após '___' do nome, minúscula e sem extensão."""
    stem = Path(filename).stem
    return stem.split("___", 1)[-1].lower().strip()


def load_leaf_map(root: Path = PLANTVILLAGE_DIR) -> dict[str, str]:
    path = root / "leaf-map.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v[0] for k, v in raw.items() if v}


def list_records(root: Path, use_leaf_groups: bool = True) -> list[ImageRecord]:
    """Lista as imagens de `root/<classe>/` e associa cada uma a um grupo (folha)."""
    leaf_map = load_leaf_map(root) if use_leaf_groups else {}
    records: list[ImageRecord] = []
    for cls in CLASS_NAMES:
        folder = root / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            leaf = leaf_map.get(_leaf_key(p.name))
            group = f"{cls}:{leaf}" if leaf else f"{cls}:{p.name}"
            records.append(ImageRecord(path=p, label=cls, group=group))
    return records


def dhash(img: Image.Image, size: int = 8) -> str:
    """Hash perceptual de diferença (dHash) 8x8 → 64 bits em hexadecimal."""
    small = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def clean_records(records: list[ImageRecord]) -> tuple[list[ImageRecord], CleaningReport]:
    """Remove arquivos corrompidos, duplicatas exatas (MD5) e quase-duplicatas (dHash idêntico)."""
    report = CleaningReport(total_listed=len(records))
    seen_md5: set[str] = set()
    seen_dhash: set[tuple[str, str]] = set()
    kept: list[ImageRecord] = []
    for rec in records:
        try:
            data = rec.path.read_bytes()
            with Image.open(rec.path) as im:
                im.load()
                rgb = im.convert("RGB")
                if min(rgb.size) < 32:
                    raise ValueError("imagem muito pequena")
                h = dhash(rgb)
        except Exception:
            report.corrupted_removed += 1
            continue
        md5 = hashlib.md5(data).hexdigest()
        if md5 in seen_md5:
            report.exact_duplicates_removed += 1
            continue
        if (rec.label, h) in seen_dhash:
            report.near_duplicates_removed += 1
            continue
        seen_md5.add(md5)
        seen_dhash.add((rec.label, h))
        kept.append(rec)
    report.valid_unique_images = len(kept)
    for cls in CLASS_NAMES:
        report.per_class_after_cleaning[cls] = sum(1 for r in kept if r.label == cls)
    return kept, report


def split_70_15_15(records: list[ImageRecord], seed: int = 42) -> dict[str, np.ndarray]:
    """Índices de treino/validação/teste (≈71/14/14%), estratificado por classe e agrupado por folha."""
    labels = np.array([r.label for r in records])
    groups = np.array([r.group for r in records])
    idx = np.arange(len(records))

    sgkf1 = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest, test = next(sgkf1.split(idx, labels, groups))
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgkf2.split(rest, labels[rest], groups[rest]))
    return {"train": rest[tr_rel], "val": rest[va_rel], "test": test}


def verify_no_group_leakage(records: list[ImageRecord], split: dict[str, np.ndarray]) -> dict[str, int]:
    """Conta grupos (folhas) compartilhados entre partições — deve ser 0."""
    sets = {k: {records[i].group for i in v} for k, v in split.items()}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


_NUM_RE = re.compile(r"\d+")


def extract_features(
    records: list[ImageRecord],
    cache_name: str,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Extrai (com cache em `results/cache/`) descritores clássicos, embeddings e métricas de qualidade."""
    from src.vision.embedder import MobileNetV2Embedder
    from src.vision.image_processor import ImageProcessor

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_name}.npz"
    paths = np.array([str(r.path.relative_to(ROOT)) for r in records])
    if cache_file.exists() and not force:
        cached = np.load(cache_file, allow_pickle=False)
        if "multiscale" in cached.files and len(cached["paths"]) == len(paths) and bool(np.all(cached["paths"] == paths)):
            return {k: cached[k] for k in cached.files}

    processor = ImageProcessor()
    embedder = MobileNetV2Embedder()
    handcrafted, embeddings, multiscale, brightness, blur = [], [], [], [], []
    for rec in records:
        proc = processor.process(rec.path)
        if proc.tensor is None or proc.feature_vector is None:
            raise RuntimeError(f"Falha ao processar {rec.path}")
        handcrafted.append(proc.feature_vector)
        ms = embedder.embed_multiscale_tensor(proc.tensor)
        multiscale.append(ms)
        embeddings.append(ms[:1280])
        brightness.append(proc.brightness)
        blur.append(proc.blur_score)
    out = {
        "paths": paths,
        "labels": np.array([r.label for r in records]),
        "handcrafted": np.vstack(handcrafted).astype(np.float32),
        "embeddings": np.vstack(embeddings).astype(np.float32),
        "multiscale": np.vstack(multiscale).astype(np.float32),
        "brightness": np.array(brightness, dtype=np.float32),
        "blur_score": np.array(blur, dtype=np.float32),
    }
    np.savez_compressed(cache_file, **out)
    return out


def augment_image(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Aumento de dados padrão (aplicado só ao treino): recorte/zoom, giro, rotação, cor e desfoque."""
    from PIL import ImageEnhance, ImageFilter

    w, h = img.size
    scale = rng.uniform(0.5, 1.0)
    cw, ch = int(w * np.sqrt(scale)), int(h * np.sqrt(scale))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img = img.rotate(float(rng.uniform(-25, 25)), resample=Image.Resampling.BILINEAR, fillcolor=(int(rng.integers(60, 200)),) * 3)
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enh(img).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    return img


def extract_augmented_features(records: list[ImageRecord], cache_name: str, n_copies: int = 2, seed: int = 42) -> dict[str, np.ndarray]:
    """Embeddings multiescala de `n_copies` versões aumentadas por imagem (cache em `results/cache/`)."""
    from src.vision.embedder import MobileNetV2Embedder

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_name}.npz"
    paths = np.array([str(r.path.relative_to(ROOT)) for r in records])
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=False)
        if len(cached["paths"]) == len(paths) * n_copies:
            return {k: cached[k] for k in cached.files}
    embedder = MobileNetV2Embedder()
    rng = np.random.default_rng(seed)
    ms, lab, pth = [], [], []
    for rec in records:
        with Image.open(rec.path) as im:
            base = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        for _ in range(n_copies):
            arr = np.asarray(augment_image(base, rng), dtype=np.uint8)
            ms.append(embedder.embed_multiscale(arr))
            lab.append(rec.label)
            pth.append(str(rec.path.relative_to(ROOT)))
    out = {"paths": np.array(pth), "labels": np.array(lab), "multiscale": np.vstack(ms).astype(np.float32)}
    np.savez_compressed(cache_file, **out)
    return out
