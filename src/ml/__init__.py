"""Módulo de Machine Learning: Treinamento, Divisão Estratificada e Avaliação Científica."""

from src.ml.evaluation import ExperimentEvaluator, ModelEvaluationReport
from src.ml.training import DatasetSplitSummary, VisualModelTrainer

__all__ = [
    "DatasetSplitSummary",
    "VisualModelTrainer",
    "ExperimentEvaluator",
    "ModelEvaluationReport",
]

