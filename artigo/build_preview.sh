#!/usr/bin/env bash
# Pré-visualização local do artigo SEM a classe IEEEtran (que o ambiente de desenvolvimento não tem).
# A versão oficial deve ser compilada no Overleaf com main.tex (IEEEtran, modo conference).
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import re
s = open("main.tex", encoding="utf-8").read()
s = s.replace(r"\documentclass[conference]{IEEEtran}", r"""\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[top=0.75in,bottom=1in,left=0.625in,right=0.625in,columnsep=0.25in]{geometry}
\newcommand{\IEEEauthorblockN}[1]{#1}\newcommand{\IEEEauthorblockA}[1]{\\\small #1}
\newenvironment{IEEEkeywords}{\par\small\textbf{\textit{Palavras-chave}---}}{\par}
\renewcommand{\abstractname}{Resumo}
\renewcommand{\refname}{Referências}""")
s = s.replace(r"\usepackage[brazil]{babel}", "")
s = s.replace(r"\usepackage{microtype}", "")
open("main_preview.tex", "w", encoding="utf-8").write(s)
PY
for _ in 1 2 3; do pdflatex -interaction=nonstopmode main_preview.tex >/dev/null; done
mv main_preview.pdf PlantAI_artigo_preview.pdf
rm -f main_preview.* *.aux *.log *.out
pdfinfo PlantAI_artigo_preview.pdf | grep Pages
