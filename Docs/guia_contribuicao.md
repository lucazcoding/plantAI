# Guia de Contribuição — PlantAI

Guia prático para contribuir com o repositório sem risco de quebrar o pipeline de
modelos. Foca em testes, documentação e curadoria de dados — atividades isoladas e
de baixo impacto.

## 1. Preparação do ambiente

```bash
git clone https://github.com/lucazcoding/plantAI.git
cd plantAI
python -m venv .venv
# Windows (PowerShell)
.venv\\Scripts\\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Antes de qualquer alteração

Rode a suíte de testes para confirmar que o repositório está saudável:

```bash
python -m pytest -q
```

## 3. Convenções do repositório

### Branches

| Tipo | Prefixo | Exemplo |
|---|---|---|
| Documentação | `docs/` | `docs/guia-contribuicao` |
| Testes | `test/` | `test/integridade-dados` |
| Correção | `fix/` | `fix/typo-readme` |

### Mensagens de commit (Conventional Commits)

```text
<tipo>: <descrição curta no imperativo>
```

Tipos usados: `docs`, `test`, `fix`, `chore`. Mescle sempre com `--no-ff` para
preservar o histórico das branches.

## 4. Onde contribuir com baixo risco

| Área | Pasta | Exemplos de tarefa |
|---|---|---|
| Testes | `tests/` | Novos testes unitários, casos de borda |
| Dados | `data/` | Validar/revisar `symptoms_sample.csv` e `question_intents.csv` |
| Documentação | `Docs/` | Explicar módulos, atualizar metodologia e resultados |

> **Evite** alterar `src/` (agentes, regras de decisão, modelos) sem revisão do Tech Lead:
> essas pastas afetam os resultados científicos do artigo.

## 5. Executando apenas os testes de dados

```bash
python -m pytest tests/test_data_exemplos.py -q
```

Os testes verificam cabeçalhos, arquivos não vazios, campos obrigatórios,
IDs numéricos e únicos, correspondência entre cultura e prefixo da classe
PlantVillage e o conjunto permitido de intenções.

## 6. Enviando a contribuição

```bash
git checkout -b docs/marco-contribuicoes
git add CONTRIBUTORS.md Docs/guia_contribuicao.md
git commit -m "docs: adiciona lista de contribuidores e guia de contribuicao"
git add tests/test_data_exemplos.py
git commit -m "test: adiciona testes de integridade dos CSVs em data/"
git push -u origin docs/marco-contribuicoes
```

Abra um Pull Request descrevendo **o que** mudou e **por que**.
