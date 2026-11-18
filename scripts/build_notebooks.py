"""Gera e EXECUTA os 8 notebooks (saídas reais embutidas). Uso: python scripts/build_notebooks.py

Os notebooks recalculam métricas a partir dos dados/características reais (nada de mocks) e exibem os
artefatos de `results/`. Requer `bash scripts/download_datasets.sh` e `python -m src.ml.run_all_experiments`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
M = ROOT / 'results' / 'metrics'
def show(name): display(Image(filename=str(ROOT / 'results' / name), width=620))
"""

NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_exploracao_dataset.ipynb"] = [
    md("# 01 — Exploração dos datasets reais\n\n**PlantVillage** (laboratório, Mohanty et al. 2016) e **PlantDoc** (campo, Singh et al. 2020), restritos às 6 classes do PlantAI; mais o corpus textual de sintomas e o de perguntas."),
    code(SETUP),
    code("""from src.ml import dataset as ds
pv = ds.list_records(ds.PLANTVILLAGE_DIR); pdoc = ds.list_records(ds.PLANTDOC_DIR, use_leaf_groups=False)
df = pd.DataFrame({'PlantVillage': pd.Series([r.label for r in pv]).value_counts(), 'PlantDoc': pd.Series([r.label for r in pdoc]).value_counts()}).fillna(0).astype(int)
display(df); print('Total PlantVillage:', len(pv), '| Total PlantDoc:', len(pdoc))"""),
    code("show('plots/leaf_samples_plantvillage.png'); show('plots/leaf_samples_plantdoc.png')"),
    code("""sym = pd.read_csv(ROOT / 'data' / 'symptoms_sample.csv'); q = pd.read_csv(ROOT / 'data' / 'question_intents.csv')
print('Textos de sintomas:', len(sym), '| perguntas:', len(q))
display(sym.groupby(['crop','disease_class']).size().rename('textos').reset_index())
display(sym.sample(6, random_state=1)[['disease_class','symptom_description']])"""),
]

NOTEBOOKS["02_preprocessamento.ipynb"] = [
    md("# 02 — Limpeza, divisão sem vazamento e extração de características\n\n1. Integridade e duplicatas (MD5 + dHash). 2. Divisão 71/14/14% **estratificada por classe e agrupada por folha** (PlantVillage fotografa a mesma folha várias vezes). 3. Descritores clássicos (30) e embeddings MobileNetV2 (1984-d multiescala)."),
    code(SETUP),
    code("print(json.dumps(json.load(open(M/'dataset_split_70_15_15.json')), indent=1, ensure_ascii=False)[:2500]); show('plots/dataset_class_distribution.png')"),
    code("""from src.vision.image_processor import ImageProcessor
proc = ImageProcessor(); res = proc.process(ROOT/'results'/'exemplos'/'plantvillage_Tomato_Early_blight.jpg')
print('válida:', res.is_valid, '| brilho: %.1f | nitidez: %.1f' % (res.brightness, res.blur_score), '| padrões:', res.detected_visual_patterns)
print('vetor de descritores:', res.feature_vector.shape, '| tensor:', res.tensor.shape, res.tensor.min(), res.tensor.max())"""),
    code("""# Qualidade das imagens REAIS (mesmos limiares das regras de decisão)
f = np.load(ROOT/'results'/'cache'/'plantvillage_features.npz'); g = np.load(ROOT/'results'/'cache'/'plantdoc_features.npz')
for name, d in (('PlantVillage', f), ('PlantDoc', g)):
    print(f"{name}: brilho médio={d['brightness'].mean():.1f} | nitidez mediana={np.median(d['blur_score']):.1f} | rejeitadas por qualidade={(d['blur_score']<18).sum()+(d['brightness']<25).sum()} / {len(d['blur_score'])}")"""),
]

NOTEBOOKS["03_treinamento_cnn.ipynb"] = [
    md("# 03 — Experimento 1: baselines visuais\n\nClasse majoritária e regressão logística sobre 6 estatísticas RGB — o piso de desempenho antes do MobileNetV2."),
    code(SETUP),
    code("""from src.ml import dataset as ds
from src.ml.run_all_experiments import make_head
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
recs, _ = ds.clean_records(ds.list_records(ds.PLANTVILLAGE_DIR)); sp = ds.split_70_15_15(recs)
F = ds.extract_features(recs, 'plantvillage_features'); y = F['labels']; hc = F['handcrafted']
tr, te = sp['train'], sp['test']
maj = DummyClassifier(strategy='most_frequent').fit(hc[tr], y[tr])
base = make_head('logreg', c=0.03).fit(hc[tr][:, :6], y[tr])
print('Majoritária  acc = %.4f' % (maj.predict(hc[te]) == y[te]).mean())
print('6 RGB+LogReg acc = %.4f' % (base.predict(hc[te][:, :6]) == y[te]).mean())
print(classification_report(y[te], base.predict(hc[te][:, :6]), digits=3))"""),
    code("show('confusion_matrices/exp1_baseline_vision_cm.png')"),
]

NOTEBOOKS["04_transfer_learning.ipynb"] = [
    md("# 04 — Experimentos 2 e 3: MobileNetV2 pré-treinado + cabeça, e hiperparâmetros\n\nBackbone ImageNet **congelado** (TFLite INT8) → características → cabeça supervisionada treinada no PlantVillage. Hiperparâmetros escolhidos na **validação**; o teste só é relatado ao final."),
    code(SETUP),
    code("display(pd.DataFrame(json.load(open(M/'exp2_feature_sets.json')))); display(pd.DataFrame(json.load(open(M/'exp3_hyperparameters.json'))))"),
    code("show('plots/exp2_model_comparison.png'); show('plots/exp3_hyperparameters.png'); show('plots/exp2_learning_curve.png')"),
    code("""# Recalcula ao vivo a cabeça logística (C=0.1) sobre as características em cache e confere com o JSON salvo
from src.ml import dataset as ds
from src.ml.run_all_experiments import make_head
from sklearn.metrics import accuracy_score, f1_score
recs, _ = ds.clean_records(ds.list_records(ds.PLANTVILLAGE_DIR)); sp = ds.split_70_15_15(recs)
F = ds.extract_features(recs, 'plantvillage_features'); y = F['labels']
m = make_head('logreg', c=0.1).fit(F['multiscale'][sp['train']], y[sp['train']]); p = m.predict(F['multiscale'][sp['test']])
print('Teste: acc=%.4f  F1-macro=%.4f' % (accuracy_score(y[sp['test']], p), f1_score(y[sp['test']], p, average='macro')))"""),
    code("show('confusion_matrices/exp2_transfer_learning_cm.png')"),
]

NOTEBOOKS["05_experimentos_pln.ipynb"] = [
    md("# 05 — Experimento 4: PLN\n\nClassificação de sintomas (TF-IDF + LR vs. LinearSVC calibrado) com validação cruzada 5-fold e **reconhecimento de intenção da pergunta** do usuário."),
    code(SETUP),
    code("""from src.nlp.symptom_classifier import SymptomClassifier
from src.nlp.question_intent import QuestionIntentClassifier, split_user_message
for algo in ('logistic_regression', 'svm'):
    m = SymptomClassifier(algorithm=algo).train_from_csv(evaluate_cv=True)
    print(f'{algo:20s} acc={m.accuracy:.4f} F1={m.f1_macro:.4f} (N={m.num_samples}, 5-fold)')
print('Intenção da pergunta (CV 5-fold):', QuestionIntentClassifier().evaluate_cv())"""),
    code("show('confusion_matrices/exp4_nlp_lr_cm.png'); show('confusion_matrices/exp4_nlp_svm_cm.png')"),
    code("""clf = SymptomClassifier(); clf.train_from_csv(evaluate_cv=False); qi = QuestionIntentClassifier().train_from_csv()
for msg in ['Folhas de tomateiro com anéis concêntricos tipo alvo e halo amarelo. Como tratar?',
            'Minha batata ficou com a folha preta e molhada depois da chuva, tem mofo branco. É grave?',
            'Folha de tomate verde e limpa, sem manchas. Está saudável?']:
    parts = split_user_message(msg); out = clf.predict(parts.symptom_text); intent = qi.predict(parts.question)
    print(msg); print('  ->', out['predicted_class'], f"({out['confidence']:.0%})", '| sintomas:', out['metadata']['extracted_symptoms'], '| intenção:', intent[0])"""),
    md("**Limitação:** o corpus foi redigido pela equipe (180 textos); textos sem a cultura citada são ambíguos entre tomate e batata — é aí que a imagem complementa o texto (Exp 5)."),
]

NOTEBOOKS["06_avaliacao_modelos.ipynb"] = [
    md("# 06 — Avaliação consolidada e generalização de domínio (Exp 6)"),
    code(SETUP),
    code("display(pd.read_csv(M/'summary_all_experiments.csv')); print(json.dumps(json.load(open(M/'environment.json')), indent=1))"),
    code("""e6 = json.load(open(M/'exp6_generalization_plantdoc.json')); e6b = json.load(open(M/'exp6_domain_adaptation.json'))
print('PlantDoc sem adaptação : acc=%.4f IC95=%s' % (e6['accuracy'], [round(v,3) for v in e6['extra_metadata']['accuracy_ci95']]))
print('confiança média PlantDoc=%.3f (acurácia %.3f) -> probabilidades não calibradas em campo' % (e6['extra_metadata']['mean_confidence_plantdoc'], e6['accuracy']))
print('Adaptação (50%% PlantDoc no treino): acc=%.4f vs %.4f sem adaptação, mesmo subconjunto' % (e6b['accuracy'], e6b['extra_metadata']['same_subset_without_adaptation_acc']))"""),
    code("show('plots/exp6_generalization_comparison.png'); show('confusion_matrices/exp6_generalization_plantdoc_cm.png')"),
]

NOTEBOOKS["07_integracao_multiagente.ipynb"] = [
    md("# 07 — Integração: imagem + pergunta → visão + PLN + regras (Exp 5)\n\nTodos os resultados vêm do modelo real; não há probabilidades digitadas."),
    code(SETUP),
    code("""from src.agents import CoordinatorAgent
co = CoordinatorAgent(); EX = ROOT / 'results' / 'exemplos'
for img, msg in [('plantvillage_Tomato_Late_blight.jpg', 'Folhas encharcadas escuras com mofo branco embaixo. Como tratar?'),
                 ('plantvillage_Tomato_healthy.jpg', 'Pústulas marrom-avermelhadas cor de ferrugem nas folhas de milho. Qual doença é essa?'),
                 ('plantdoc_Potato_Late_blight.jpg', 'Como evitar isso?')]:
    r = co.answer(EX / img, msg)
    print(img, '|', msg); print('  intenção:', r.intent, '| status:', r.decision.status.value, '| classe:', r.decision.final_class); print('  ', r.answer[:240], '\\n')"""),
    code("e5 = json.load(open(M/'exp5_multiagent_ablation.json'))\nrows=[]\nfor ds_name in ('plantvillage_test','plantdoc_field'):\n    for mode in ('vision_only','nlp_only','integrated'):\n        r = e5[ds_name]['headline_p0'][mode]; rows.append({'conjunto':ds_name,'modo':mode, **{k:r[k] for k in ('accuracy','coverage','accuracy_when_answered','f1_macro')}})\ndisplay(pd.DataFrame(rows))"),
    code("show('plots/exp5_multiagent_ablation.png'); show('plots/exp5_sensitivity.png'); show('confusion_matrices/exp5_multiagent_integrated_cm.png')"),
    md("**Leitura honesta:** a acurácia *estrita* do integrado é menor que a da visão em PlantVillage porque divergências viram `INCONCLUSIVO` (o sistema se abstém). A vantagem é a **precisão quando responde** e a detecção de erro (principalmente em campo). Ver Docs/experiments.md."),
]

NOTEBOOKS["00_PlantAI_Projeto_Completo.ipynb"] = [
    md("# PlantAI — Notebook mestre\n\nAFYA | UNIMA Maceió — Inteligência Artificial. Resumo executável: dados reais → ML/Visão → PLN → decisão → integração.\n\n| Critério | Evidência |\n|---|---|\n| Problema e arquitetura | `Docs/architecture.md`, `src/contracts.py` |\n| Machine Learning | Exp 1–3 (`src/ml/`) |\n| Visão Computacional | `src/vision/` (processador + MobileNetV2) |\n| PLN | `src/nlp/` (sintomas + intenção) |\n| Decisão | `src/decision/decision_rules.py` (6 regras) |\n| Integração | `CoordinatorAgent.answer` |\n| Experimentos | `results/`, `Docs/experiments.md` |"),
    code(SETUP),
    code("display(pd.read_csv(M/'summary_all_experiments.csv')[['experiment_id','dataset_name','num_samples','accuracy','f1_macro']])"),
    code("""from src.agents import CoordinatorAgent
co = CoordinatorAgent()
r = co.answer(ROOT/'results'/'exemplos'/'plantvillage_Potato_Late_blight.jpg', 'Folhas pretas e encharcadas com mofo branco após a chuva. Isso é grave?')
print(co.generate_report(r.decision)); print('\\nRESPOSTA:', r.answer)"""),
    code("show('plots/exp5_multiagent_ablation.png'); show('plots/exp6_generalization_comparison.png')"),
]


def main() -> None:
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(nb, NB / name)
        print("ok", name)


if __name__ == "__main__":
    main()
