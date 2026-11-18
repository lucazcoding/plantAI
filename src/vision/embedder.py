"""Extrator de características profundas: MobileNetV2 pré-treinado (ImageNet).

Usa o modelo MobileNetV2 1.0/224 quantizado (INT8, TensorFlow Lite) distribuído em
`google-coral/test_data` (Apache-2.0) e versionado em `models/`. O vetor de 1280
dimensões vem da camada `Logits/AvgPool` (global average pooling), exatamente a
representação que a "cabeça" de transfer learning do MobileNetV2 consome.

Por que TFLite e não Keras? O ambiente de execução do projeto (Python 3.13, sem GPU)
não possui TensorFlow completo; o runtime `ai-edge-litert` roda o mesmo grafo na CPU
(~7 ms/imagem) e é instalável via pip. O fine-tuning completo em Keras continua
disponível em `scripts/colab_finetune_mobilenetv2.py` (ver README).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mobilenet_v2_1.0_224_quant.tflite"
EMBEDDING_DIM = 1280
# Blocos residuais intermediários (resoluções 28x28, 14x14 e 7x7) cujas ativações são resumidas por média e
# desvio-padrão espaciais: capturam textura de lesões em várias escalas, além do vetor global final.
MULTISCALE_BLOCKS = (5, 9, 12, 15)
MULTISCALE_DIM = EMBEDDING_DIM + 2 * (32 + 64 + 96 + 160)  # 1984


class MobileNetV2Embedder:
    """Gera embeddings de 1280 dimensões a partir de imagens RGB 224x224 (uint8)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, num_threads: int = 2) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pesos MobileNetV2 não encontrados em {self.model_path}. "
                "Veja models/README.md para obter o arquivo."
            )
        self._interp = Interpreter(
            model_path=str(self.model_path),
            experimental_preserve_all_tensors=True,
            num_threads=num_threads,
        )
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        pool = [d for d in self._interp.get_tensor_details() if d["name"].endswith("Logits/AvgPool")]
        if not pool:
            raise RuntimeError("Tensor 'Logits/AvgPool' não encontrado no modelo TFLite.")
        self._pool_index = int(pool[0]["index"])
        self._pool_scale, self._pool_zero = pool[0]["quantization"]
        details = {d["name"]: d for d in self._interp.get_tensor_details()}
        self._blocks = []
        for n in MULTISCALE_BLOCKS:
            d = details[f"MobilenetV2/expanded_conv_{n}/add"]
            self._blocks.append((int(d["index"]), *d["quantization"]))

    def embed(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Embedding (1280,) de uma imagem uint8 (224, 224, 3)."""
        if rgb_uint8.shape != (224, 224, 3) or rgb_uint8.dtype != np.uint8:
            raise ValueError(f"Esperado uint8 (224, 224, 3); recebido {rgb_uint8.dtype} {rgb_uint8.shape}.")
        self._interp.set_tensor(self._input["index"], rgb_uint8[np.newaxis, ...])
        self._interp.invoke()
        raw = self._interp.get_tensor(self._pool_index).reshape(-1).astype(np.float32)
        return (raw - self._pool_zero) * self._pool_scale

    def embed_tensor(self, tensor_01: np.ndarray) -> np.ndarray:
        """Embedding a partir do tensor normalizado [0,1] (1, 224, 224, 3) do `ImageProcessor`."""
        arr = np.clip(np.rint(tensor_01[0] * 255.0), 0, 255).astype(np.uint8)
        return self.embed(arr)

    def embed_multiscale(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Vetor multiescala (1984-d): embedding global final + média/desvio de 4 blocos intermediários."""
        final = self.embed(rgb_uint8)
        parts = [final]
        means, stds = [], []
        for idx, scale, zero in self._blocks:
            act = (self._interp.get_tensor(idx).astype(np.float32) - zero) * scale  # (1, H, W, C)
            flat = act.reshape(-1, act.shape[-1])
            means.append(flat.mean(axis=0))
            stds.append(flat.std(axis=0))
        parts += means + stds
        return np.concatenate(parts).astype(np.float32)

    def embed_multiscale_tensor(self, tensor_01: np.ndarray) -> np.ndarray:
        arr = np.clip(np.rint(tensor_01[0] * 255.0), 0, 255).astype(np.uint8)
        return self.embed_multiscale(arr)
