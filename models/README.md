# Diretório de Modelos Treinados (`models/`)

Conforme a Seção 8 da documentação oficial do projeto (`Sistema-MultiAgent-PlantVillage.mdown`), os arquivos binários pesados de pesos neurais (`.keras`, `.h5`, `.pt`) não devem ser versionados diretamente no Git para não exceder os limites do repositório.

## 1. Como Obter ou Gerar os Modelos

### 1.1. Execução Local Reprodutível (Imediata)
Ao executar `python demo_pipeline.py` ou os notebooks da pasta `notebooks/`, o sistema:
- Treina e avalia automaticamente os classificadores de PLN (`TF-IDF + LogisticRegression` e `TF-IDF + LinearSVC Calibrado`) usando `data/symptoms_sample.csv`.
- Treina e avalia a cabeça de classificação visual sobre os 30 descritores extraídos pelo `ImageProcessor` (`src/ml/training.py`).

### 1.2. Pesos MobileNetV2 (Google Colab + Google Drive)
Durante o treinamento com GPU no Google Colab (`notebooks/04_transfer_learning.ipynb`):
1. O modelo `PlantAI_MobileNetV2` é construído via `MobileNetV2Classifier.build_mobilenetv2_architecture(num_classes=6)`.
2. Os pesos treinados são exportados para `models/mobilenetv2_plantvillage.keras` (ou montados via Google Drive).
3. Para carregar os pesos no `VisionAgent` / `MLAgent`:
   ```python
   from src.agents import VisionAgent
   agent = VisionAgent(weights_path="models/mobilenetv2_plantvillage.keras")
   ```
   Caso o arquivo `.keras` não esteja presente localmente, o classificador opera automaticamente com o modelo de atributos visuais treinado ou em modo fallback determinístico, permitindo que todos os testes (`pytest`) e fluxos multiagente rodem sem interrupção.

