"""Gera, a partir de results/metrics/*.json, os números e tabelas usados no artigo e na documentação.

Evita transcrição manual: `artigo/numeros.tex` (macros), `artigo/tabelas.tex` e `Docs/results_tables.md`.
Uso: python scripts/make_report_assets.py   (depois de python -m src.ml.run_all_experiments)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"
J = lambda n: json.loads((M / f"{n}.json").read_text(encoding="utf-8"))  # noqa: E731
pct = lambda v, d=1: f"{100 * v:.{d}f}".replace(".", ",")  # noqa: E731  (vírgula decimal, padrão pt-BR)
CLS = {"Tomato___Early_blight": "Tomate — pinta-preta", "Tomato___Late_blight": "Tomate — requeima", "Tomato___healthy": "Tomate — saudável",
       "Potato___Early_blight": "Batata — pinta-preta", "Potato___Late_blight": "Batata — requeima", "Corn_(maize)___Common_rust_": "Milho — ferrugem-comum"}


def main() -> None:
    sp, e1, e1m, e2a, e2, e6, e6b = (J(n) for n in ("dataset_split_70_15_15", "exp1_baseline_vision", "exp1_baseline_majority", "exp2_classic_descriptors", "exp2_transfer_learning", "exp6_generalization_plantdoc", "exp6_domain_adaptation"))
    fs, hp, e5 = J("exp2_feature_sets"), J("exp3_hyperparameters"), J("exp5_multiagent_ablation")
    lr, svm, qi = J("exp4_nlp_lr"), J("exp4_nlp_svm"), J("exp4_question_intent")
    abl = J("exp4_nlp_crop_boost_ablation")
    env = J("environment")
    ci = lambda r: f"[{pct(r['extra_metadata']['accuracy_ci95'][0])}; {pct(r['extra_metadata']['accuracy_ci95'][1])}]"  # noqa: E731
    final = next(r for r in fs if r["feature_set"] == "final")
    pdc = json.loads((ROOT / "results" / "metrics" / "dataset_split_70_15_15.json").read_text())  # noqa: F841

    # ---------- macros
    mac: dict[str, str] = {
        "nTotal": str(sp["cleaning"]["total_listed"]), "nUnique": str(sp["cleaning"]["valid_unique_images"]), "nDup": str(sp["cleaning"]["exact_duplicates_removed"]),
        "nTrain": str(sp["train_count"]), "nVal": str(sp["val_count"]), "nTest": str(sp["test_count"]), "nGroups": str(sp["unique_groups"]), "nGrouped": str(sp["images_with_leaf_group"]),
        "nPlantDoc": str(e6["num_samples"]), "nSym": str(lr["num_samples"]),
        "accMajority": pct(e1m["accuracy"]), "accRGB": pct(e1["accuracy"]), "accClassic": pct(e2a["accuracy"]), "accFinal": pct(final["test_accuracy"]), "accMobile": pct(e2["accuracy"]), "fMobile": pct(e2["f1_macro"]),
        "ciMobile": ci(e2), "selectedHead": env["selected_head"].replace("_", r"\_").replace("%", r"\%"),
        "nlpLRpure": pct(abl["logreg_pure"]["accuracy"]), "nlpSVMpure": pct(abl["svm_pure"]["accuracy"]), "nlpLR": pct(lr["accuracy"]), "nlpSVM": pct(svm["accuracy"]), "nlpLRci": ci(lr), "nlpSVMci": ci(svm), "intentAcc": pct(qi["accuracy"]), "nIntent": str(qi["n"]),
        "pdAcc": pct(e6["accuracy"]), "pdCI": ci(e6), "pdConf": pct(e6["extra_metadata"]["mean_confidence_plantdoc"]), "pvConf": pct(e6["extra_metadata"]["mean_confidence_plantvillage_test"]),
        "pdHalfNo": pct(e6b["extra_metadata"]["same_subset_without_adaptation_acc"]), "pdHalfYes": pct(e6b["accuracy"]), "nPDhalf": str(e6b["num_samples"]),
    }
    for ds, tag in (("plantvillage_test", "PV"), ("plantdoc_field", "PD")):
        h = e5[ds]["headline_p0"]
        mac[f"n{tag}"] = str(h["n"])
        for mode, m_tag in (("vision_only", "Vis"), ("nlp_only", "Nlp"), ("integrated", "Int")):
            mac[f"acc{m_tag}{tag}"], mac[f"cov{m_tag}{tag}"], mac[f"sel{m_tag}{tag}"] = pct(h[mode]["accuracy"]), pct(h[mode]["coverage"]), pct(h[mode]["accuracy_when_answered"])
        for s in e5[ds]["sensitivity_to_wrong_user_text"]:
            if s["p_wrong_text"] in (0.1, 0.3):
                suf = {0.1: "Ten", 0.3: "Thirty"}[s["p_wrong_text"]]  # nomes de macro LaTeX não aceitam dígitos
                mac[f"sensInt{tag}{suf}"] = pct(s["integrated"]["accuracy"])
                mac[f"sensSel{tag}{suf}"] = pct(s["integrated"]["accuracy_when_answered"])
        st = h["statuses_integrated"]
        mac[f"divPct{tag}"] = pct(st.get("DIVERGENCIA_DETECTADA", 0) / h["n"])
    (ROOT / "artigo" / "numeros.tex").write_text("% GERADO por scripts/make_report_assets.py — não editar\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n", encoding="utf-8")

    # ---------- tabelas (LaTeX + Markdown)
    rows_vis = [("Classe majoritária", e1m), ("6 estatísticas RGB + LogReg", e1), ("30 descritores clássicos + MLP", e2a), ("MobileNetV2 (1280-d final) + LogReg", None), (f"MobileNetV2 multiescala + cabeça (implantado)", e2)]
    tex = ["% GERADO por scripts/make_report_assets.py", "\\begin{table}[t]\\caption{Modelos visuais no conjunto de teste do PlantVillage (N=" + str(sp["test_count"]) + "). IC 95\\% por bootstrap.}\\label{tab:vis}\\centering\\small",
           "\\begin{tabular}{lcc}\\toprule Modelo & Acurácia (\\%) & F1-macro (\\%)\\\\\\midrule"]
    md = ["| Modelo | Acurácia (%) | F1-macro (%) | IC95% acurácia |", "|---|--:|--:|---|"]
    for name, r in rows_vis:
        if r is None:
            acc, f1, c = final["test_accuracy"], final["test_f1_macro"], "—"
        else:
            acc, f1, c = r["accuracy"], r["f1_macro"], ci(r) if r["extra_metadata"].get("accuracy_ci95") else "—"
        tex.append(f"{name} & {pct(acc)} & {pct(f1)}\\\\")
        md.append(f"| {name} | {pct(acc)} | {pct(f1)} | {c} |")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")

    tex += ["\\begin{table}[t]\\caption{PLN: validação cruzada 5-fold (textos avaliados fora da dobra). ``Puro'' = TF-IDF + modelo linear; ``regra de cultura'' = reforço $\\times$1,35/$\\times$0,75 do NLPAgent.}\\label{tab:nlp}\\centering\\small",
            "\\begin{tabular}{lccc}\\toprule Modelo & N & Acurácia (\\%) & F1-macro (\\%)\\\\\\midrule",
            f"TF-IDF + LogReg (puro) & {lr['num_samples']} & {pct(abl['logreg_pure']['accuracy'])} & {pct(abl['logreg_pure']['f1_macro'])}\\\\",
            f"TF-IDF + SVC calibrado (puro) & {svm['num_samples']} & {pct(abl['svm_pure']['accuracy'])} & {pct(abl['svm_pure']['f1_macro'])}\\\\",
            f"NLPAgent: LogReg + regra de cultura & {lr['num_samples']} & {pct(lr['accuracy'])} & {pct(lr['f1_macro'])}\\\\",
            f"NLPAgent: SVC + regra de cultura & {svm['num_samples']} & {pct(svm['accuracy'])} & {pct(svm['f1_macro'])}\\\\",
            f"Intenção da pergunta (LogReg) & {qi['n']} & {pct(qi['accuracy'])} & {pct(qi['f1_macro'])}\\\\\\bottomrule\\end{{tabular}}\\end{{table}}"]
    md += ["", "| Modelo PLN | N | Acurácia (%) | F1-macro (%) |", "|---|--:|--:|--:|",
           f"| TF-IDF + LogReg (puro, sem regra de cultura) | {lr['num_samples']} | {pct(abl['logreg_pure']['accuracy'])} | {pct(abl['logreg_pure']['f1_macro'])} |",
           f"| TF-IDF + LinearSVC calibrado (puro, sem regra de cultura) | {svm['num_samples']} | {pct(abl['svm_pure']['accuracy'])} | {pct(abl['svm_pure']['f1_macro'])} |",
           f"| NLPAgent: TF-IDF + LogReg + reforço de cultura (implantado) | {lr['num_samples']} | {pct(lr['accuracy'])} | {pct(lr['f1_macro'])} |",
           f"| NLPAgent: TF-IDF + LinearSVC + reforço de cultura | {svm['num_samples']} | {pct(svm['accuracy'])} | {pct(svm['f1_macro'])} |",
           f"| Intenção da pergunta | {qi['n']} | {pct(qi['accuracy'])} | {pct(qi['f1_macro'])} |"]

    tex += ["\\begin{table*}[t]\\caption{Experimento 5 — ablação (texto correto). Acurácia estrita conta \\textit{inconclusivo} como erro; cobertura = fração respondida; precisão = acurácia entre as respondidas.}\\label{tab:abl}\\centering\\small",
            "\\begin{tabular}{llccc}\\toprule Conjunto & Modo & Acurácia estrita (\\%) & Cobertura (\\%) & Precisão se respondeu (\\%)\\\\\\midrule"]
    md += ["", "| Conjunto | Modo | Acurácia estrita (%) | Cobertura (%) | Precisão se respondeu (%) | IC95% acurácia |", "|---|---|--:|--:|--:|---|"]
    for ds, label in (("plantvillage_test", f"PlantVillage teste (N={e5['plantvillage_test']['headline_p0']['n']})"), ("plantdoc_field", f"PlantDoc campo (N={e5['plantdoc_field']['headline_p0']['n']})")):
        for mode, ml in (("vision_only", "Visão isolada"), ("nlp_only", "PLN isolado"), ("integrated", "Integrado (visão+PLN+regras)")):
            r = e5[ds]["headline_p0"][mode]
            tex.append(f"{label} & {ml} & {pct(r['accuracy'])} & {pct(r['coverage'])} & {pct(r['accuracy_when_answered'])}\\\\")
            md.append(f"| {label} | {ml} | {pct(r['accuracy'])} | {pct(r['coverage'])} | {pct(r['accuracy_when_answered'])} | [{pct(r['ci95'][0])}; {pct(r['ci95'][1])}] |")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule\\end{tabular}\\end{table*}"
    md += ["", "Sensibilidade ao erro do usuário (fração de relatos textuais errados → acurácia estrita / precisão do integrado):", "", "| Conjunto | p=0,1 | p=0,2 | p=0,3 | p=0,5 |", "|---|---|---|---|---|"]
    for ds in ("plantvillage_test", "plantdoc_field"):
        md.append(f"| {ds} | " + " | ".join(f"{pct(s['integrated']['accuracy'])} / {pct(s['integrated']['accuracy_when_answered'])}" for s in e5[ds]["sensitivity_to_wrong_user_text"]) + " |")

    tex += ["\\begin{table}[t]\\caption{Generalização de domínio (acurácia, \\%).}\\label{tab:dom}\\centering\\small",
            "\\begin{tabular}{lc}\\toprule Cenário & Acurácia\\\\\\midrule",
            f"PlantVillage, teste (N={sp['test_count']}) & {pct(e2['accuracy'])}\\\\",
            f"PlantDoc todo (N={e6['num_samples']}) & {pct(e6['accuracy'])}\\\\",
            f"PlantDoc 50\\% (N={e6b['num_samples']}), sem adapt. & {pct(e6b['extra_metadata']['same_subset_without_adaptation_acc'])}\\\\",
            f"PlantDoc 50\\%, com adapt. & {pct(e6b['accuracy'])}\\\\\\bottomrule\\end{{tabular}}\\end{{table}}"]
    md += ["", "| Cenário | Acurácia (%) |", "|---|--:|", f"| PlantVillage teste | {pct(e2['accuracy'])} |", f"| PlantDoc inteiro, sem adaptação | {pct(e6['accuracy'])} {ci(e6)} |",
           f"| PlantDoc 50% reservado, sem adaptação | {pct(e6b['extra_metadata']['same_subset_without_adaptation_acc'])} |", f"| PlantDoc 50% reservado, com adaptação | {pct(e6b['accuracy'])} {ci(e6b)} |"]

    md += ["", "Conjuntos de características (cabeça LogReg C=0,1):", "", "| Conjunto | Dim | F1 val | Acc teste | Acc PlantDoc |", "|---|--:|--:|--:|--:|"]
    md += [f"| {r['feature_set']} | {r['dim']} | {pct(r['val_f1_macro'])} | {pct(r['test_accuracy'])} | {pct(r['plantdoc_accuracy'])} |" for r in fs]
    md += ["", "Hiperparâmetros da cabeça (seleção pela validação):", "", "| Configuração | F1 val | F1 teste | Acc PlantDoc |", "|---|--:|--:|--:|"]
    md += [f"| {r['config_name']}{' ✔ (melhor na validação)' if r.get('selected_by_validation') else ''}{' ✔ (implantada)' if r.get('deployed') else ''} | {pct(r['val_f1_macro'])} | {pct(r['test_f1_macro'])} | {pct(r['plantdoc_accuracy'])} |" for r in hp]
    md += ["", "Divisão: " + ", ".join(f"{k}={sp[k + '_count']}" for k in ("train", "val", "test")) + f"; folhas (grupos)={sp['unique_groups']}; imagens com folha mapeada={sp['images_with_leaf_group']}/{sp['cleaning']['valid_unique_images']}."]
    (ROOT / "artigo" / "tabelas.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (ROOT / "Docs" / "results_tables.md").write_text("# Tabelas de resultados (geradas automaticamente)\n\nGeradas por `scripts/make_report_assets.py` a partir de `results/metrics/*.json`.\n\n" + "\n".join(md) + "\n", encoding="utf-8")

    # ---------- figuras do artigo (cópias) e diagrama da arquitetura
    import shutil
    for src, dst in (("confusion_matrices/exp2_transfer_learning_cm.png", "cm_visao.png"), ("plots/exp5_multiagent_ablation.png", "ablacao.png"), ("plots/exp6_generalization_comparison.png", "dominio.png"),
                     ("plots/leaf_samples_plantdoc.png", "amostras_plantdoc.png"), ("plots/exp5_sensitivity.png", "sensibilidade.png")):
        shutil.copy(ROOT / "results" / src, ROOT / "artigo" / "figuras" / dst)
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    ax.axis("off")
    boxes = {"img": (0.01, 0.55, 0.14, 0.35, "Foto da\nfolha"), "msg": (0.01, 0.05, 0.14, 0.35, "Mensagem\n(sintomas +\npergunta)"),
             "vis": (0.24, 0.55, 0.22, 0.35, "VisionAgent\nImageProcessor\n+ MobileNetV2\n+ cabeça (MLAgent)"), "nlp": (0.24, 0.05, 0.22, 0.35, "NLPAgent\nTF-IDF + LogReg\n+ intenção da\npergunta"),
             "dec": (0.55, 0.30, 0.2, 0.4, "DecisionRulesEngine\n6 regras:\nqualidade, fallback,\nconcordância,\ndesempate, divergência,\nincerteza"), "out": (0.83, 0.30, 0.16, 0.4, "Resposta +\nrelatório +\nhistórico")}
    for k, (x, y, w, h, t) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc="#e8f0fe" if k in ("vis", "nlp") else "#fff4e5" if k == "dec" else "#eef7ee", ec="#333"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.5)
    for (x1, y1, x2, y2) in ((0.15, 0.72, 0.24, 0.72), (0.15, 0.22, 0.24, 0.22), (0.46, 0.72, 0.55, 0.55), (0.46, 0.22, 0.55, 0.45), (0.75, 0.5, 0.83, 0.5)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(0.5, 0.97, "CoordinatorAgent (orquestra o fluxo e registra o histórico)", ha="center", fontsize=7, style="italic")
    fig.savefig(ROOT / "artigo" / "figuras" / "arquitetura.png", bbox_inches="tight")
    shutil.copy(ROOT / "artigo" / "figuras" / "arquitetura.png", ROOT / "results" / "plots" / "arquitetura.png")
    print("ok:", len(mac), "macros")
    _ = re  # noqa


if __name__ == "__main__":
    main()
