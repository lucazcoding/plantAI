"""Testes unitários do classificador visual, divisão 70/15/15 e módulo de ML (`src/ml/`)."""

from __future__ import annotations

from pathlib import Path

from src.agents.ml_agent import MLAgent
from src.ml.evaluation import ExperimentEvaluator
from src.ml.training import VisualModelTrainer
from src.vision.image_processor import ImageProcessor


def test_stratified_split_70_15_15_and_deduplication() -> None:
    """Verifica se a divisão 70/15/15 é estratificada e remove corrompidas e duplicatas."""
    trainer = VisualModelTrainer()
    prepared = trainer.build_and_split_dataset(
        samples_per_class=20,
        inject_corrupted=3,
        inject_duplicates=5,
    )
    summary = prepared.summary
    assert summary.corrupted_removed == 3
    assert summary.duplicates_removed == 5
    assert summary.valid_unique_images == 120  # 6 classes * 20 amostras
    assert summary.train_count == 84          # 70% de 120
    assert summary.val_count == 18            # 15% de 120
    assert summary.test_count == 18           # 15% de 120

    # Verifica estratificação exata no conjunto de teste (3 amostras por classe)
    for count in summary.class_distribution_test.values():
        assert count == 3


def test_baseline_vs_transfer_learning_head_and_ml_agent(tmp_path: Path) -> None:
    """Garante que o modelo de representação completa supera o baseline e integra ao MLAgent."""
    trainer = VisualModelTrainer()
    dataset = trainer.build_and_split_dataset(
        samples_per_class=30,
        inject_corrupted=0,
        inject_duplicates=0,
    )

    baseline_model, _ = trainer.train_baseline_model(dataset)
    tl_model, _, history = trainer.train_transfer_learning_head(dataset, max_iter=20)

    evaluator = ExperimentEvaluator(results_root=tmp_path / "results")
    base_preds = list(baseline_model.predict(dataset.x_test[:, :2]))
    tl_preds = list(tl_model.predict(dataset.x_test))

    rep_base = evaluator.evaluate_predictions(
        experiment_id="test_exp1",
        model_name="Baseline RGB",
        dataset_name="SyntheticPlantVillage",
        y_true=dataset.y_test,
        y_pred=base_preds,
        class_names=trainer.class_names,
    )
    rep_tl = evaluator.evaluate_predictions(
        experiment_id="test_exp2",
        model_name="Transfer Learning Head",
        dataset_name="SyntheticPlantVillage",
        y_true=dataset.y_test,
        y_pred=tl_preds,
        class_names=trainer.class_names,
    )

    assert rep_tl.accuracy >= rep_base.accuracy
    assert rep_tl.accuracy >= 0.85
    assert len(history["train_acc"]) == 20

    cm_file = evaluator.save_confusion_matrix_plot(rep_tl)
    assert cm_file.exists()

    # Integra o classificador treinado ao MLAgent e testa predição em imagem real
    ml_agent = MLAgent()
    ml_agent.fit_from_features(dataset.x_train, dataset.y_train, max_iter=150)
    proc_img = ImageProcessor().process(dataset.test_images[0])
    out = ml_agent.classify_processed_image(proc_img)
    assert out["agent_type"] == "vision"
    assert out["metadata"]["ml_agent"] == "MLAgent"
    assert out["metadata"]["inference_backend"] == "trained_visual_mlp_head"
