"""Executa TODOS os experimentos do PlantAI sobre dados REAIS e grava métricas, gráficos e modelo.

    python -m src.ml.run_all_experiments            # requer data/plantvillage e data/plantdoc
    (antes: bash scripts/download_datasets.sh)

Experimentos
  Exp 1  Baselines visuais: classe majoritária e regressão logística sobre 6 estatísticas RGB.
  Exp 2  Visão clássica (30 descritores + MLP) vs. MobileNetV2 pré-treinado + cabeça (transfer learning).
  Exp 3  Ajuste de hiperparâmetros da cabeça (seleção na VALIDAÇÃO; teste só para relato final).
  Exp 4  PLN: TF-IDF+LR vs TF-IDF+LinearSVC calibrado (CV 5-fold) e reconhecimento de intenção da pergunta.
  Exp 5  Integração Visão+PLN+Regras: ablação (visão, PLN, integrado) em PlantVillage-teste e PlantDoc,
         com texto avaliado "fora da dobra" (out-of-fold) e varredura de erro do usuário.
  Exp 6  Generalização de domínio (PlantVillage -> PlantDoc, fotos de campo) e adaptação de domínio.
Todas as semente = 42. Nada é simulado: imagens, rótulos e predições vêm de arquivos e modelos reais.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.contracts import AgentOutputDict
from src.decision.decision_rules import DecisionRulesEngine
from src.ml import dataset as ds
from src.ml.evaluation import ExperimentEvaluator
from src.nlp.question_intent import QuestionIntentClassifier
from src.nlp.symptom_classifier import SymptomClassifier
from src.vision.classifier import MobileNetV2Classifier

ROOT = ds.ROOT
RESULTS_DIR = ROOT / "results"
SEED = 42
INCONCLUSIVE = "INCONCLUSIVO"


# ----------------------------------------------------------------------------- utilidades
def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """Intervalo de confiança de 95% (bootstrap percentil) para a acurácia."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def short(c: str) -> str:
    return c.replace("Corn_(maize)___", "Corn_").replace("___", "\n").replace("_", " ")


def make_head(kind: str, **kw: Any) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(C=kw["c"], max_iter=3000, random_state=SEED)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=kw["layers"], alpha=kw["alpha"], learning_rate_init=kw["lr"],
            early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=SEED,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def timed_fit(model: Pipeline, x: np.ndarray, y: np.ndarray) -> float:
    t0 = time.perf_counter()
    model.fit(x, y)
    return time.perf_counter() - t0


def vision_output_from_probs(clf: MobileNetV2Classifier, probs: np.ndarray, classes: list[str], meta: dict[str, Any]) -> AgentOutputDict:
    return clf.build_output({c: float(p) for c, p in zip(classes, probs, strict=True)}, meta)


def quality_meta(brightness: float, blur: float, engine_proc: Any) -> dict[str, Any]:
    """Metadados de qualidade equivalentes aos do ImageProcessor (mesmos limiares)."""
    dark = brightness < engine_proc.min_brightness
    bright = brightness > engine_proc.max_brightness
    blurry = blur < engine_proc.min_blur_score
    return {
        "image_valid": not (dark or bright or blurry), "is_corrupted": False, "is_too_dark": bool(dark),
        "is_too_bright": bool(bright), "is_blurry": bool(blurry), "brightness": float(brightness), "blur_score": float(blur),
        "quality_issues": [m for f, m in ((dark, "Imagem extremamente escura."), (bright, "Imagem superexposta."), (blurry, "Imagem borrada.")) if f],
    }


def save_bar(path: Path, labels: list[str], series: dict[str, list[float]], title: str, ylabel: str = "Métrica", ylim: tuple[float, float] = (0, 1.08)) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 4.4), dpi=140)
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w, label=name)
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals, strict=True):
            ax.text(xi, v + 0.01, f"{v:.1%}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sample_grid(records_by_class: dict[str, list[Path]], path: Path, title: str, n_cols: int = 4) -> None:
    classes = list(records_by_class)
    fig, axes = plt.subplots(len(classes), n_cols, figsize=(2.2 * n_cols, 2.2 * len(classes)), dpi=120)
    for r, cls in enumerate(classes):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(records_by_class[cls]):
                ax.imshow(Image.open(records_by_class[cls][c]).convert("RGB").resize((160, 160)))
            if c == 0:
                ax.set_title(short(cls).replace("\n", " "), fontsize=7, loc="left")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- pipeline
def run_all_scientific_experiments() -> dict[str, Any]:
    if not ds.dataset_available() or not ds.dataset_available(ds.PLANTDOC_DIR):
        raise SystemExit("Datasets ausentes. Execute: bash scripts/download_datasets.sh")

    ev = ExperimentEvaluator(RESULTS_DIR)
    classes = ds.CLASS_NAMES
    out: dict[str, Any] = {}

    # ===== Dados: limpeza, divisão agrupada por folha, extração de características
    records, clean = ds.clean_records(ds.list_records(ds.PLANTVILLAGE_DIR))
    split = ds.split_70_15_15(records, seed=SEED)
    leak = ds.verify_no_group_leakage(records, split)
    feats = ds.extract_features(records, "plantvillage_features")
    labels = feats["labels"]
    emb, hc = feats["embeddings"], feats["handcrafted"]
    idx_tr, idx_va, idx_te = split["train"], split["val"], split["test"]
    mapped = sum(1 for r in records if not r.group.endswith((".JPG", ".jpg", ".jpeg", ".png")))

    def part(i: np.ndarray) -> dict[str, int]:
        v, c = np.unique(labels[i], return_counts=True)
        return {str(a): int(b) for a, b in zip(v, c, strict=True)}

    split_summary = {
        "dataset": "PlantVillage (color) — Hughes & Salathé 2015 / Mohanty et al. 2016, 6 classes",
        "cleaning": clean.to_dict(),
        "images_with_leaf_group": mapped,
        "unique_groups": len({r.group for r in records}),
        "train_count": len(idx_tr), "val_count": len(idx_va), "test_count": len(idx_te),
        "proportions": {k: round(len(v) / len(records), 4) for k, v in split.items()},
        "class_distribution_train": part(idx_tr), "class_distribution_val": part(idx_va), "class_distribution_test": part(idx_te),
        "group_leakage_between_partitions": leak,
        "split_method": "StratifiedGroupKFold (classe estratificada, folha agrupada), seed=42",
    }
    (ev.metrics_dir / "dataset_split_70_15_15.json").write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (ev.metrics_dir / "split_manifest.json").write_text(
        json.dumps({k: [str(records[i].path.relative_to(ROOT)) for i in v] for k, v in split.items()}, ensure_ascii=False), encoding="utf-8")
    out["split"] = split_summary

    # distribuição das partições
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    xp, w = np.arange(len(classes)), 0.26
    for k, (name, i) in enumerate((("Treino", idx_tr), ("Validação", idx_va), ("Teste", idx_te))):
        d = part(i)
        ax.bar(xp + (k - 1) * w, [d.get(c, 0) for c in classes], width=w, label=name)
    ax.set_xticks(xp)
    ax.set_xticklabels([short(c) for c in classes], fontsize=8)
    ax.set_ylabel("Imagens")
    ax.set_title("PlantVillage real: partições 71/14/14% (estratificadas, agrupadas por folha)", fontweight="bold", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ev.plots_dir / "dataset_class_distribution.png")
    plt.close(fig)

    # PlantDoc (domínio externo; nunca usado no treino principal)
    pd_records, pd_clean = ds.clean_records(ds.list_records(ds.PLANTDOC_DIR, use_leaf_groups=False))
    pd_feats = ds.extract_features(pd_records, "plantdoc_features")
    pd_labels, pd_emb, pd_hc = pd_feats["labels"], pd_feats["embeddings"], pd_feats["handcrafted"]
    out["plantdoc_cleaning"] = pd_clean.to_dict()

    rng_s = np.random.default_rng(SEED)
    sample_grid(
        {c: [records[i].path for i in rng_s.choice(np.where(labels == c)[0], 4, replace=False)] for c in classes},
        ev.plots_dir / "leaf_samples_plantvillage.png", "PlantVillage (imagens reais, laboratório)")
    sample_grid(
        {c: [pd_records[i].path for i in rng_s.choice(np.where(pd_labels == c)[0], 4, replace=False)] for c in classes},
        ev.plots_dir / "leaf_samples_plantdoc.png", "PlantDoc (imagens reais, campo)")

    # ===== Exp 1 — baselines
    ci = lambda yt, yp: bootstrap_ci((np.asarray(yt) == np.asarray(yp)).astype(float))  # noqa: E731
    dummy = DummyClassifier(strategy="most_frequent").fit(hc[idx_tr], labels[idx_tr])
    p_dummy = dummy.predict(hc[idx_te])
    rep_dummy = ev.evaluate_predictions("exp1_baseline_majority", "Exp 1A — Baseline: classe majoritária", "PlantVillage (teste)",
                                         list(labels[idx_te]), list(p_dummy), classes)
    base = make_head("logreg", c=0.03)
    t_base = timed_fit(base, hc[idx_tr][:, :6], labels[idx_tr])
    p_base = base.predict(hc[idx_te][:, :6])
    rep1 = ev.evaluate_predictions("exp1_baseline_vision", "Exp 1B — Baseline: 6 estatísticas RGB + LogReg", "PlantVillage (teste)",
                                    list(labels[idx_te]), list(p_base), classes, t_base,
                                    {"features_used": 6, "split": "71/14/14", "accuracy_ci95": ci(labels[idx_te], p_base)})
    ev.save_confusion_matrix_plot(rep1, "exp1_baseline_vision_cm.png", cmap="Oranges")

    # ===== Exp 2 — visão clássica vs transfer learning (e conjuntos de características)
    classic = make_head("mlp", layers=(128, 64), alpha=1e-3, lr=1e-3)
    t_classic = timed_fit(classic, hc[idx_tr], labels[idx_tr])
    p_classic = classic.predict(hc[idx_te])
    rep2a = ev.evaluate_predictions("exp2_classic_descriptors", "Exp 2A — 30 descritores clássicos + MLP", "PlantVillage (teste)",
                                     list(labels[idx_te]), list(p_classic), classes, t_classic,
                                     {"features_used": 30, "accuracy_ci95": ci(labels[idx_te], p_classic)})

    ms, pd_ms = feats["multiscale"], pd_feats["multiscale"]
    feature_sets = {
        "final": (emb, pd_emb),
        "multiscale": (ms, pd_ms),
        "multiscale+handcrafted": (np.hstack([ms, hc]), np.hstack([pd_ms, pd_hc])),
    }
    fs_rows = []
    for fs_name, (xa, xp) in feature_sets.items():
        m = make_head("logreg", c=0.1)
        m.fit(xa[idx_tr], labels[idx_tr])
        pv, pt = m.predict(xa[idx_va]), m.predict(xa[idx_te])
        fs_rows.append({"feature_set": fs_name, "dim": int(xa.shape[1]), "head": "LogReg C=0.1",
                        "val_f1_macro": round(float(f1_score(labels[idx_va], pv, average="macro")), 4),
                        "test_accuracy": round(float(accuracy_score(labels[idx_te], pt)), 4),
                        "test_f1_macro": round(float(f1_score(labels[idx_te], pt, average="macro")), 4),
                        "plantdoc_accuracy": round(float(accuracy_score(pd_labels, m.predict(xp))), 4)})
    # Seleção NA VALIDAÇÃO entre as duas representações do MobileNetV2 (a variante +handcrafted é só ablação)
    fs_name = max(("final", "multiscale"), key=lambda k: next(r["val_f1_macro"] for r in fs_rows if r["feature_set"] == k))
    for r in fs_rows:
        r["deployed_representation"] = r["feature_set"] == fs_name
    (ev.metrics_dir / "exp2_feature_sets.json").write_text(json.dumps(fs_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    emb_sel, pd_emb_sel = feature_sets[fs_name]

    # ===== Exp 3 — hiperparâmetros da cabeça (seleção na validação)
    configs: list[dict[str, Any]] = [{"kind": "logreg", "c": c, "name": f"LogReg C={c:g}"} for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)]
    configs += [
        {"kind": "mlp", "layers": (256,), "alpha": 1e-3, "lr": 1e-3, "name": "MLP (256) lr=1e-3"},
        {"kind": "mlp", "layers": (512, 128), "alpha": 1e-2, "lr": 1e-3, "name": "MLP (512,128) lr=1e-3"},
        {"kind": "mlp", "layers": (256,), "alpha": 1e-3, "lr": 5e-2, "name": "MLP (256) lr=5e-2 (instável)"},
    ]
    hp: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for cfg in configs:
        m = make_head(cfg["kind"], **{k: v for k, v in cfg.items() if k not in ("kind", "name")})
        t = timed_fit(m, emb_sel[idx_tr], labels[idx_tr])
        pv, pt = m.predict(emb_sel[idx_va]), m.predict(emb_sel[idx_te])
        hp.append({"config_name": cfg["name"], "val_accuracy": round(accuracy_score(labels[idx_va], pv), 4),
                   "val_f1_macro": round(f1_score(labels[idx_va], pv, average="macro"), 4),
                   "test_accuracy": round(accuracy_score(labels[idx_te], pt), 4),
                   "test_f1_macro": round(f1_score(labels[idx_te], pt, average="macro"), 4), "training_time_sec": round(t, 3)})
        fitted[cfg["name"]] = m
    best = max(hp, key=lambda r: (r["val_f1_macro"], r["val_accuracy"]))
    best_cfg = next(c for c in configs if c["name"] == best["config_name"])

    # Variante com aumento de dados (só no treino): regra a priori — adota-se a cabeça aumentada se a
    # validação (PlantVillage) ficar a até 1 p.p. de F1 da melhor sem aumento (prior de robustez a campo).
    aug_feats = ds.extract_augmented_features([records[i] for i in idx_tr], "plantvillage_train_aug", n_copies=2)
    aug_x = aug_feats["multiscale"] if fs_name == "multiscale" else aug_feats["multiscale"][:, :1280]
    aug_head = make_head(best_cfg["kind"], **{k: v for k, v in best_cfg.items() if k not in ("kind", "name")})
    aug_head.fit(np.vstack([emb_sel[idx_tr], aug_x]), np.concatenate([labels[idx_tr], aug_feats["labels"]]))
    pv_aug, pt_aug = aug_head.predict(emb_sel[idx_va]), aug_head.predict(emb_sel[idx_te])
    aug_row = {"config_name": best["config_name"] + " + aumento de dados (2 cópias)",
               "val_accuracy": round(accuracy_score(labels[idx_va], pv_aug), 4), "val_f1_macro": round(f1_score(labels[idx_va], pv_aug, average="macro"), 4),
               "test_accuracy": round(accuracy_score(labels[idx_te], pt_aug), 4), "test_f1_macro": round(f1_score(labels[idx_te], pt_aug, average="macro"), 4),
               "plantdoc_accuracy": round(float(accuracy_score(pd_labels, aug_head.predict(pd_emb_sel))), 4)}
    use_aug = aug_row["val_f1_macro"] >= best["val_f1_macro"] - 0.01
    for r in hp:
        r["selected_by_validation"] = r is best
        r["plantdoc_accuracy"] = round(float(accuracy_score(pd_labels, fitted[r["config_name"]].predict(pd_emb_sel))), 4)
    aug_row["deployed"] = use_aug
    hp.append(aug_row)
    (ev.metrics_dir / "exp3_hyperparameters.json").write_text(json.dumps(hp, indent=2, ensure_ascii=False), encoding="utf-8")
    save_bar(ev.plots_dir / "exp3_hyperparameters.png", [r["config_name"].replace(" (", "\n(").replace(" + ", "\n+ ") for r in hp],
             {"F1-macro (validação)": [r["val_f1_macro"] for r in hp], "F1-macro (teste)": [r["test_f1_macro"] for r in hp]},
             f"Exp 3 — Hiperparâmetros da cabeça (características: {fs_name})", "F1-macro", (0.5, 1.05))

    final_head = aug_head if use_aug else fitted[best["config_name"]]
    final_name = aug_row["config_name"] if use_aug else best["config_name"]
    p_te = final_head.predict(emb_sel[idx_te])
    proba_te = final_head.predict_proba(emb_sel[idx_te])
    rep2b = ev.evaluate_predictions("exp2_transfer_learning", f"Exp 2B — MobileNetV2 (ImageNet) + cabeça [{final_name}]",
                                     "PlantVillage (teste)", list(labels[idx_te]), list(p_te), classes, best["training_time_sec"],
                                     {"features_used": int(emb_sel.shape[1]), "feature_set": fs_name, "selected_config": final_name, "accuracy_ci95": ci(labels[idx_te], p_te)})
    ev.save_confusion_matrix_plot(rep2b, "exp2_transfer_learning_cm.png", cmap="Blues")

    clf_final = MobileNetV2Classifier(weights_path=None)
    clf_final.head = final_head
    clf_final.feature_mode = fs_name
    clf_final.class_names = list(final_head.classes_)
    clf_final.head_description = f"StandardScaler + {final_name} sobre características MobileNetV2 '{fs_name}' (treino PlantVillage)"
    clf_final.save_head()

    # curva de aprendizado (acurácia no teste vs. tamanho do treino) com a cabeça logística
    fracs = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    lc = []
    for f in fracs:
        sub = idx_tr if f == 1.0 else train_test_split(idx_tr, train_size=f, stratify=labels[idx_tr], random_state=SEED)[0]
        m = make_head("logreg", c=0.1)
        m.fit(emb_sel[sub], labels[sub])
        lc.append({"train_fraction": f, "n_train": int(len(sub)), "test_accuracy": round(accuracy_score(labels[idx_te], m.predict(emb_sel[idx_te])), 4)})
    (ev.metrics_dir / "exp2_learning_curve.json").write_text(json.dumps(lc, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=140)
    ax.plot([r["n_train"] for r in lc], [r["test_accuracy"] for r in lc], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Imagens de treino (escala log)")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title("Exp 2 — Curva de aprendizado (cabeça sobre MobileNetV2)", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ev.plots_dir / "exp2_learning_curve.png")
    plt.close(fig)
    save_bar(ev.plots_dir / "exp2_model_comparison.png", ["Majoritária", "6 RGB\n+LogReg", "30 descritores\n+MLP", "MobileNetV2\n(final 1280)", "MobileNetV2\nmultiescala + cabeça"],
             {"Accuracy": [rep_dummy.accuracy, rep1.accuracy, rep2a.accuracy, next(r["test_accuracy"] for r in fs_rows if r["feature_set"] == "final"), rep2b.accuracy],
              "F1-macro": [rep_dummy.f1_macro, rep1.f1_macro, rep2a.f1_macro, next(r["test_f1_macro"] for r in fs_rows if r["feature_set"] == "final"), rep2b.f1_macro]},
             "Exp 1–2 — Modelos visuais no teste do PlantVillage")

    # ===== Exp 4 — PLN
    df_nlp = pd.read_csv(ROOT / "data" / "symptoms_sample.csv")
    texts, y_nlp = df_nlp["symptom_description"].astype(str).tolist(), df_nlp["disease_class"].astype(str).tolist()
    nlp_reports: dict[str, Any] = {}
    oof_outputs: dict[str, list[AgentOutputDict]] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for algo, tag, cmap in (("logistic_regression", "lr", "Greens"), ("svm", "svm", "Purples")):
        pred_oof: list[str] = [""] * len(texts)
        outs: list[AgentOutputDict | None] = [None] * len(texts)
        t0 = time.perf_counter()
        for tr, te in skf.split(texts, y_nlp):
            c = SymptomClassifier(algorithm=algo).train([texts[i] for i in tr], [y_nlp[i] for i in tr])  # type: ignore[arg-type]
            for i in te:
                o = c.predict(texts[i])
                outs[i] = o
                pred_oof[i] = o["predicted_class"]
        el = time.perf_counter() - t0
        rep = ev.evaluate_predictions(f"exp4_nlp_{tag}", f"Exp 4{'A' if tag == 'lr' else 'B'} — PLN TF-IDF + {'LogisticRegression' if tag == 'lr' else 'LinearSVC calibrado'}",
                                       f"Corpus de sintomas (CV 5-fold fora-da-dobra, N={len(texts)})", y_nlp, pred_oof, classes, el,
                                       {"accuracy_ci95": ci(y_nlp, pred_oof)})
        ev.save_confusion_matrix_plot(rep, f"exp4_nlp_{tag}_cm.png", cmap=cmap)
        nlp_reports[tag] = rep
        oof_outputs[tag] = outs  # type: ignore[assignment]
    qi = QuestionIntentClassifier().evaluate_cv()
    (ev.metrics_dir / "exp4_question_intent.json").write_text(json.dumps(qi, indent=2), encoding="utf-8")
    (ev.metrics_dir / "exp4_nlp_models.json").write_text(json.dumps({"logreg": nlp_reports["lr"].to_dict(), "svm": nlp_reports["svm"].to_dict(), "question_intent": qi}, indent=2, ensure_ascii=False), encoding="utf-8")
    # ambiguidade cultura-agnóstica: textos tomate/batata
    out["exp4"] = {"lr": nlp_reports["lr"].to_dict(), "svm": nlp_reports["svm"].to_dict(), "intent": qi}
    # ablação: TF-IDF puro vs. NLPAgent com reforço por cultura citada (mesmo CV 5-fold)
    from src.ml import nlp_ablation
    nlp_abl = nlp_ablation.run(seed=SEED)
    (ev.metrics_dir / "exp4_nlp_crop_boost_ablation.json").write_text(json.dumps(nlp_abl, indent=2, ensure_ascii=False), encoding="utf-8")
    out["exp4"]["crop_boost_ablation"] = nlp_abl

    # ===== Exp 5 — integração (ablação) com PLN fora-da-dobra
    engine = DecisionRulesEngine()
    from src.vision.image_processor import ImageProcessor
    proc = ImageProcessor()
    nlp_by_class: dict[str, list[AgentOutputDict]] = {c: [] for c in classes}
    for i, o in enumerate(oof_outputs["lr"]):
        nlp_by_class[y_nlp[i]].append(o)

    def fusion_eval(y: np.ndarray, probs: np.ndarray, brightness: np.ndarray, blur: np.ndarray, p_wrong: float, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        res = {"vision_only": [], "nlp_only": [], "integrated": []}
        statuses: dict[str, int] = {}
        for k in range(len(y)):
            v_out = vision_output_from_probs(clf_final, probs[k], list(final_head.classes_), quality_meta(float(brightness[k]), float(blur[k]), proc))
            text_cls = y[k]
            if rng.random() < p_wrong:
                text_cls = rng.choice([c for c in classes if c != y[k]])
            n_out = nlp_by_class[text_cls][rng.integers(0, len(nlp_by_class[text_cls]))]
            for mode in res:
                d = engine.evaluate(vision_output=v_out if mode != "nlp_only" else None, nlp_output=n_out if mode != "vision_only" else None, mode=mode)  # type: ignore[arg-type]
                res[mode].append(d.final_class or INCONCLUSIVE)
                if mode == "integrated":
                    statuses[d.status.value] = statuses.get(d.status.value, 0) + 1
        summary: dict[str, Any] = {"p_wrong_text": p_wrong, "n": int(len(y)), "statuses_integrated": statuses}
        for mode, preds in res.items():
            pr = np.array(preds)
            correct = pr == y
            answered = pr != INCONCLUSIVE
            lo, hi = bootstrap_ci(correct.astype(float))
            summary[mode] = {"accuracy": round(float(correct.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                             "f1_macro": round(float(f1_score(y, pr, labels=classes, average="macro", zero_division=0)), 4),
                             "coverage": round(float(answered.mean()), 4),
                             "accuracy_when_answered": round(float(correct[answered].mean()) if answered.any() else 0.0, 4)}
        summary["_preds"] = {m: list(map(str, v)) for m, v in res.items()}
        return summary

    probs_pv = proba_te
    probs_pd = final_head.predict_proba(pd_emb_sel)
    exp5: dict[str, Any] = {"nlp_source": "predições fora-da-dobra (5-fold) do TF-IDF+LR; cada texto nunca foi visto pelo modelo que o classificou"}
    for name, y, pr, br, bl in (
        ("plantvillage_test", labels[idx_te], probs_pv, feats["brightness"][idx_te], feats["blur_score"][idx_te]),
        ("plantdoc_field", pd_labels, probs_pd, pd_feats["brightness"], pd_feats["blur_score"]),
    ):
        sweep = [fusion_eval(y, pr, br, bl, p, SEED + int(p * 100)) for p in (0.0, 0.1, 0.2, 0.3, 0.5)]
        main = sweep[0]
        preds_int = main.pop("_preds")
        for s in sweep[1:]:
            s.pop("_preds")
        exp5[name] = {"headline_p0": main, "sensitivity_to_wrong_user_text": sweep[1:]}
        rep = ev.evaluate_predictions(f"exp5_integrated_{name}", f"Exp 5 — Sistema integrado ({name})", name, list(y), preds_int["integrated"], classes + [INCONCLUSIVE][:0])
        if name == "plantvillage_test":
            ev.save_confusion_matrix_plot(rep, "exp5_multiagent_integrated_cm.png", cmap="YlGnBu")
    (ev.metrics_dir / "exp5_multiagent_ablation.json").write_text(json.dumps(exp5, indent=2, ensure_ascii=False), encoding="utf-8")
    out["exp5"] = exp5

    labs = ["PlantVillage\n(teste)", "PlantDoc\n(campo)"]
    save_bar(ev.plots_dir / "exp5_multiagent_ablation.png", labs,
             {"Visão isolada": [exp5["plantvillage_test"]["headline_p0"]["vision_only"]["accuracy"], exp5["plantdoc_field"]["headline_p0"]["vision_only"]["accuracy"]],
              "PLN isolado": [exp5["plantvillage_test"]["headline_p0"]["nlp_only"]["accuracy"], exp5["plantdoc_field"]["headline_p0"]["nlp_only"]["accuracy"]],
              "Integrado (visão+PLN+regras)": [exp5["plantvillage_test"]["headline_p0"]["integrated"]["accuracy"], exp5["plantdoc_field"]["headline_p0"]["integrated"]["accuracy"]]},
             "Exp 5 — Ablação (acurácia estrita; INCONCLUSIVO conta como erro)")
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=140)
    for name, lbl in (("plantvillage_test", "PlantVillage"), ("plantdoc_field", "PlantDoc")):
        pts = [exp5[name]["headline_p0"]] + exp5[name]["sensitivity_to_wrong_user_text"]
        ax.plot([p["p_wrong_text"] for p in pts], [p["integrated"]["accuracy"] for p in pts], marker="o", label=f"Integrado — {lbl}")
        ax.plot([p["p_wrong_text"] for p in pts], [p["vision_only"]["accuracy"] for p in pts], ls="--", label=f"Visão isolada — {lbl}")
    ax.set_xlabel("Fração de relatos textuais errados (simulação de erro do usuário)")
    ax.set_ylabel("Acurácia")
    ax.set_title("Exp 5 — Robustez da fusão ao erro do usuário", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ev.plots_dir / "exp5_sensitivity.png")
    plt.close(fig)

    # ===== Exp 6 — generalização de domínio e adaptação
    p_pd = final_head.predict(pd_emb_sel)
    rep6 = ev.evaluate_predictions("exp6_generalization_plantdoc", "Exp 6A — PlantVillage → PlantDoc (sem adaptação)", f"PlantDoc (campo, N={len(pd_labels)})",
                                    list(pd_labels), list(p_pd), classes, 0.0,
                                    {"plantvillage_test_acc": rep2b.accuracy, "accuracy_ci95": ci(pd_labels, p_pd),
                                     "mean_confidence_plantdoc": round(float(final_head.predict_proba(pd_emb_sel).max(axis=1).mean()), 4),
                                     "mean_confidence_plantvillage_test": round(float(proba_te.max(axis=1).mean()), 4),
                                     "note": "confiança média >> acurácia em campo: as probabilidades da cabeça não são calibradas fora do domínio"})
    ev.save_confusion_matrix_plot(rep6, "exp6_generalization_plantdoc_cm.png", cmap="Reds")
    pd_tr, pd_te = train_test_split(np.arange(len(pd_labels)), test_size=0.5, stratify=pd_labels, random_state=SEED)
    adapt = make_head("logreg", c=0.1)
    adapt.fit(np.vstack([emb_sel[idx_tr], pd_emb_sel[pd_tr]]), np.concatenate([labels[idx_tr], pd_labels[pd_tr]]))
    p_ad = adapt.predict(pd_emb_sel[pd_te])
    p_noad = final_head.predict(pd_emb_sel[pd_te])
    rep6b = ev.evaluate_predictions("exp6_domain_adaptation", "Exp 6B — Adaptação: PlantVillage + 50% do PlantDoc", f"PlantDoc (50%% reservado, N={len(pd_te)})".replace("%%", "%"),
                                     list(pd_labels[pd_te]), list(p_ad), classes, 0.0,
                                     {"same_subset_without_adaptation_acc": float(accuracy_score(pd_labels[pd_te], p_noad)), "accuracy_ci95": ci(pd_labels[pd_te], p_ad)})
    save_bar(ev.plots_dir / "exp6_generalization_comparison.png", ["PlantVillage\n(teste)", "PlantDoc\n(sem adaptação)", "PlantDoc 50%\n(sem adaptação)", "PlantDoc 50%\n(com adaptação)"],
             {"Accuracy": [rep2b.accuracy, rep6.accuracy, float(accuracy_score(pd_labels[pd_te], p_noad)), rep6b.accuracy],
              "F1-macro": [rep2b.f1_macro, rep6.f1_macro, float(f1_score(pd_labels[pd_te], p_noad, average="macro")), rep6b.f1_macro]},
             "Exp 6 — Laboratório vs. campo e adaptação de domínio")

    # ===== Tabela consolidada + ambiente
    reps = [rep_dummy, rep1, rep2a, rep2b, nlp_reports["lr"], nlp_reports["svm"], rep6, rep6b]
    df_sum = pd.DataFrame([{k: r.to_dict()[k] for k in ("experiment_id", "model_name", "dataset_name", "num_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro", "training_time_sec")} for r in reps])
    df_sum.to_csv(ev.metrics_dir / "summary_all_experiments.csv", index=False, encoding="utf-8")
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "seed": SEED,
           "backbone": "MobileNetV2 1.0/224 INT8 (TFLite) — ImageNet, google-coral/test_data (Apache-2.0)", "selected_head": final_name, "feature_set": fs_name}
    (ev.metrics_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    out.update({"exp1": rep1.to_dict(), "exp2a": rep2a.to_dict(), "exp2b": rep2b.to_dict(), "exp3": hp, "exp6": rep6.to_dict(), "exp6b": rep6b.to_dict(), "learning_curve": lc, "env": env, "exp2_feature_sets": fs_rows})
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = run_all_scientific_experiments()
    print(json.dumps({k: res[k] for k in ("exp1", "exp2a", "exp2b", "exp6", "exp6b")}, indent=1, ensure_ascii=False)[:3000])
    print(f"Concluído em {time.time() - t0:.0f}s. Resultados em results/ e modelo em models/plantai_head.joblib")
