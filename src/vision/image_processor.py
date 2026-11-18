"""Pré-processamento de imagens, extração de características visuais e validação de qualidade.

Atende à Seção 4.2 da documentação (`Sistema-MultiAgent-PlantVillage.mdown`):
1. Receber a imagem.
2. Verificar o formato e integridade (detecção de arquivos corrompidos).
3. Redimensionar a imagem para `224x224`.
4. Normalizar os valores dos pixels em `[0.0, 1.0]`.
5. Preparar os dados para o modelo de Machine Learning.
6. Extrair características visuais quantitativas (índices espectrais RGB/HSV, razão de área
   clorótica/necrótica, energia de bordas/textura e nitidez Laplaciana).
7. Identificar padrões visuais relevantes (anéis concêntricos, lesões úmidas, pústulas, folha sadia).
8. Encaminhar os resultados ao classificador (`MLAgent` / `MobileNetV2Classifier`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)
MIN_DIMENSION_PX: int = 32
DEFAULT_MIN_BRIGHTNESS: float = 25.0
DEFAULT_MAX_BRIGHTNESS: float = 245.0
DEFAULT_MIN_BLUR_SCORE: float = 18.0


@dataclass(slots=True)
class ProcessedImage:
    """Resultado do pré-processamento, controle de qualidade e extração visual de uma imagem."""

    tensor: np.ndarray | None            # Array (1, 224, 224, 3) normalizado [0.0, 1.0]
    original_size: tuple[int, int] | None
    is_valid: bool
    is_corrupted: bool
    is_too_dark: bool
    is_too_bright: bool
    is_blurry: bool
    brightness: float
    blur_score: float
    visual_features: dict[str, float] = field(default_factory=dict)
    feature_vector: np.ndarray | None = None
    detected_visual_patterns: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        """Converte métricas de qualidade e características visuais para o contrato padronizado."""
        return {
            "image_valid": self.is_valid,
            "is_corrupted": self.is_corrupted,
            "is_too_dark": self.is_too_dark,
            "is_too_bright": self.is_too_bright,
            "is_blurry": self.is_blurry,
            "brightness": round(float(self.brightness), 2),
            "blur_score": round(float(self.blur_score), 2),
            "original_size": self.original_size,
            "target_size": TARGET_IMAGE_SIZE,
            "visual_features": {k: round(float(v), 4) for k, v in self.visual_features.items()},
            "detected_visual_patterns": list(self.detected_visual_patterns),
            "quality_issues": list(self.quality_issues),
        }


class ImageProcessor:
    """Processador de Visão Computacional para inspeção, normalização e extração de atributos."""

    def __init__(
        self,
        target_size: tuple[int, int] = TARGET_IMAGE_SIZE,
        min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
        max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
        min_blur_score: float = DEFAULT_MIN_BLUR_SCORE,
    ) -> None:
        self.target_size = target_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_score = min_blur_score

    @staticmethod
    def compute_laplacian_variance(gray_image: np.ndarray) -> float:
        """Calcula a variância do Laplaciano 2D discreto para medir nitidez (foco)."""
        gray = gray_image.astype(np.float64)
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        return float(np.var(laplacian))

    @staticmethod
    def compute_brightness(rgb_array: np.ndarray) -> float:
        """Calcula o brilho médio perceptual (luminância ITU-R BT.601) na escala 0-255."""
        rgb = rgb_array.astype(np.float64)
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return float(np.mean(luminance))

    @staticmethod
    def extract_visual_features(rgb_norm: np.ndarray) -> tuple[dict[str, float], np.ndarray, list[str]]:
        """Extrai vetor de características visuais (cor, índices agronômicos e textura) da imagem [0, 1].

        Retorna:
        - `features_dict`: resumo interpretável das características extraídas.
        - `feature_vector`: vetor 1D NumPy (28 features) para uso por classificadores de ML.
        - `patterns`: padrões visuais detectados a partir dos descritores.
        """
        r = rgb_norm[..., 0]
        g = rgb_norm[..., 1]
        b = rgb_norm[..., 2]

        # Estatísticas de cor por canal (RGB)
        mean_r, std_r = float(np.mean(r)), float(np.std(r))
        mean_g, std_g = float(np.mean(g)), float(np.std(g))
        mean_b, std_b = float(np.mean(b)), float(np.std(b))

        # Índice de Excesso de Verde (ExG = 2G - R - B) usado em visão computacional agrícola
        exg = 2.0 * g - r - b
        mean_exg = float(np.mean(exg))
        std_exg = float(np.std(exg))

        # Razão de tecido saudável verde
        healthy_green_mask = (g > r * 1.12) & (g > b * 1.15) & (g > 0.25)
        green_ratio = float(np.mean(healthy_green_mask))

        # Razão de tecido necrótico escuro/marrom (típico de Early/Late blight)
        necrotic_mask = (r > g * 0.95) & (g < 0.42) & (b < 0.35) & (r > 0.15)
        necrotic_ratio = float(np.mean(necrotic_mask))

        # Razão de halo clorótico amarelado (R e G altos, B baixo)
        chlorotic_mask = (r > 0.52) & (g > 0.48) & (b < 0.32) & (np.abs(r - g) < 0.18)
        chlorotic_ratio = float(np.mean(chlorotic_mask))

        # Razão de pústulas ferruginosas alaranjadas/marrom-avermelhadas (R >> G > B)
        rust_mask = (r > g * 1.32) & (r > 0.42) & (b < 0.28)
        rust_ratio = float(np.mean(rust_mask))

        # Razão de lesões encharcadas escuras/úmidas nas bordas (baixa luminância esverdeada/azulada)
        water_soaked_mask = (r < 0.28) & (g < 0.34) & (b > 0.14) & (g >= r * 0.95)
        water_soaked_ratio = float(np.mean(water_soaked_mask))

        # Gradientes espaciais (energia de bordas e textura de alta frequência)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        grad_energy_x = float(np.mean(np.abs(gx)))
        grad_energy_y = float(np.mean(np.abs(gy)))
        grad_std = float(np.std(gx) + np.std(gy))

        # Concentração espacial (centro vs bordas da folha)
        h, w = gray.shape
        center_slice = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_intensity = float(np.mean(center_slice))
        border_intensity = float(np.mean(gray)) - center_intensity

        # Histogramas compactos (4 bins por canal = 12 valores)
        hist_r, _ = np.histogram(r, bins=4, range=(0.0, 1.0), density=True)
        hist_g, _ = np.histogram(g, bins=4, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(b, bins=4, range=(0.0, 1.0), density=True)

        features_dict = {
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
            "excess_green_index": mean_exg,
            "green_healthy_ratio": green_ratio,
            "necrotic_lesion_ratio": necrotic_ratio,
            "chlorotic_halo_ratio": chlorotic_ratio,
            "rust_pustule_ratio": rust_ratio,
            "water_soaked_ratio": water_soaked_ratio,
            "texture_gradient_energy": (grad_energy_x + grad_energy_y) / 2.0,
            "center_vs_border_diff": border_intensity,
        }

        scalar_features = np.array(
            [
                mean_r, mean_g, mean_b,
                std_r, std_g, std_b,
                mean_exg, std_exg,
                green_ratio, necrotic_ratio, chlorotic_ratio, rust_ratio, water_soaked_ratio,
                grad_energy_x, grad_energy_y, grad_std,
                center_intensity, border_intensity,
            ],
            dtype=np.float32,
        )
        feature_vector = np.concatenate(
            [scalar_features, hist_r.astype(np.float32), hist_g.astype(np.float32), hist_b.astype(np.float32)]
        )

        patterns: list[str] = []
        if green_ratio > 0.72 and necrotic_ratio < 0.05 and rust_ratio < 0.04:
            patterns.append("lamina_foliar_verde_homogenea")
        if necrotic_ratio > 0.06 and chlorotic_ratio > 0.03:
            patterns.append("manchas_necroticas_com_halo_clorotico")
        if water_soaked_ratio > 0.06:
            patterns.append("lesoes_escuras_encharcadas")
        if rust_ratio > 0.06:
            patterns.append("pontuacoes_ferruginosas_alta_frequencia")
        if (grad_energy_x + grad_energy_y) / 2.0 > 0.08:
            patterns.append("textura_foliar_heterogenea_com_bordas_intensas")
        if not patterns:
            patterns.append("padrao_foliar_geral")

        return features_dict, feature_vector, patterns

    def _load_pil_image(self, image_input: str | Path | bytes | Image.Image | np.ndarray) -> Image.Image:
        """Converte diferentes formatos de entrada em uma instância PIL.Image RGB."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, np.ndarray):
            if image_input.ndim < 2:
                raise ValueError("Array NumPy inválido para imagem.")
            arr = image_input
            if arr.dtype != np.uint8:
                if np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            return Image.fromarray(arr).convert("RGB")

        if isinstance(image_input, (bytes, bytearray)):
            with Image.open(BytesIO(image_input)) as img:
                img.load()
                return img.convert("RGB")

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Arquivo de imagem não encontrado: {path}")
            with Image.open(path) as img:
                img.load()
                return img.convert("RGB")

        raise TypeError(f"Tipo de entrada de imagem não suportado: {type(image_input)!r}")

    def process(self, image_input: str | Path | bytes | Image.Image | np.ndarray | None) -> ProcessedImage:
        """Valida qualidade, redimensiona para 224x224, normaliza e extrai características visuais."""
        if image_input is None:
            return ProcessedImage(
                tensor=None,
                original_size=None,
                is_valid=False,
                is_corrupted=True,
                is_too_dark=False,
                is_too_bright=False,
                is_blurry=False,
                brightness=0.0,
                blur_score=0.0,
                quality_issues=["Nenhuma imagem fornecida ou arquivo ausente."],
            )

        try:
            pil_img = self._load_pil_image(image_input)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
            return ProcessedImage(
                tensor=None,
                original_size=None,
                is_valid=False,
                is_corrupted=True,
                is_too_dark=False,
                is_too_bright=False,
                is_blurry=False,
                brightness=0.0,
                blur_score=0.0,
                quality_issues=[f"Imagem corrompida ou ilegível: {exc}"],
            )

        width, height = pil_img.size
        quality_issues: list[str] = []

        if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
            quality_issues.append(
                f"Resolução muito baixa ({width}x{height}px; mínimo {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX}px)."
            )

        resized_img = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
        rgb_uint8 = np.asarray(resized_img, dtype=np.uint8)
        gray_uint8 = np.asarray(resized_img.convert("L"), dtype=np.uint8)

        brightness = self.compute_brightness(rgb_uint8)
        blur_score = self.compute_laplacian_variance(gray_uint8)

        is_too_dark = brightness < self.min_brightness
        is_too_bright = brightness > self.max_brightness
        is_blurry = blur_score < self.min_blur_score

        if is_too_dark:
            quality_issues.append(
                f"Imagem extremamente escura (brilho={brightness:.1f} < limiar={self.min_brightness:.1f})."
            )
        if is_too_bright:
            quality_issues.append(
                f"Imagem superexposta/estourada (brilho={brightness:.1f} > limiar={self.max_brightness:.1f})."
            )
        if is_blurry:
            quality_issues.append(
                f"Imagem excessivamente borrada/sem foco (nitidez={blur_score:.1f} < limiar={self.min_blur_score:.1f})."
            )

        is_valid = len(quality_issues) == 0
        rgb_norm = rgb_uint8.astype(np.float32) / 255.0
        normalized_tensor = rgb_norm[np.newaxis, ...]
        visual_features, feature_vector, patterns = self.extract_visual_features(rgb_norm)

        return ProcessedImage(
            tensor=normalized_tensor,
            original_size=(width, height),
            is_valid=is_valid,
            is_corrupted=False,
            is_too_dark=is_too_dark,
            is_too_bright=is_too_bright,
            is_blurry=is_blurry,
            brightness=brightness,
            blur_score=blur_score,
            visual_features=visual_features,
            feature_vector=feature_vector,
            detected_visual_patterns=patterns,
            quality_issues=quality_issues,
        )
