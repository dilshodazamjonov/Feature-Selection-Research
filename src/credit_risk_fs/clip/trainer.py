from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from credit_risk_fs.clip.checkpointing import save_checkpoint
from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss
from credit_risk_fs.clip.model import SemanticStatisticalContrastiveEncoder, count_trainable_parameters
from credit_risk_fs.clip.training_metrics import collapse_diagnostics, retrieval_metrics
from credit_risk_fs.clip.training_validation import (
    ClipTrainingConfig,
    TrainingDataBundle,
    false_negative_mask,
    resolve_device,
    tensors_for_pairs,
)
from credit_risk_fs.utils.io import write_json


@dataclass(frozen=True)
class SeedTrainingResult:
    seed: int
    best_epoch: int
    final_epoch: int
    early_stopping_epoch: int
    best_validation_loss: float
    best_validation_mrr: float
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    checkpoint_hash: str
    parameter_count: int
    epoch_metrics_path: Path
    representation_metrics_path: Path
    training_log_path: Path


def train_seed(
    *,
    config: ClipTrainingConfig,
    data: TrainingDataBundle,
    seed: int,
    output_dir: Path,
    config_snapshot_text: str,
    smoke_test: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    direction: str | None = None,
    batch_log_interval: int = 5,
) -> SeedTrainingResult:
    started = time.perf_counter()
    if batch_log_interval <= 0:
        raise ValueError("batch_log_interval must be positive")
    _set_seed(seed, deterministic=config.deterministic)
    device = resolve_device(config.device_policy)
    model = SemanticStatisticalContrastiveEncoder(config.model).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_text, train_stat = tensors_for_pairs(data.train_pairs, data.training_text, data.training_stat)
    val_text, val_stat = tensors_for_pairs(data.validation_pairs, data.training_text, data.training_stat)
    expected_shapes = {
        "train_text": (len(data.train_pairs), config.model.text_input_dim),
        "train_statistical": (
            len(data.train_pairs),
            config.model.statistical_input_dim,
        ),
        "validation_text": (
            len(data.validation_pairs),
            config.model.text_input_dim,
        ),
        "validation_statistical": (
            len(data.validation_pairs),
            config.model.statistical_input_dim,
        ),
    }
    observed_shapes = {
        "train_text": tuple(train_text.shape),
        "train_statistical": tuple(train_stat.shape),
        "validation_text": tuple(val_text.shape),
        "validation_statistical": tuple(val_stat.shape),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            "CLIP tensor adapter shape mismatch before training: "
            f"expected={expected_shapes}, observed={observed_shapes}"
        )
    if not all(
        torch.isfinite(values).all().item()
        for values in (train_text, train_stat, val_text, val_stat)
    ):
        raise RuntimeError("CLIP input tensors contain NaN/Inf before training")
    train_text = train_text.to(device)
    train_stat = train_stat.to(device)
    val_text = val_text.to(device)
    val_stat = val_stat.to(device)
    train_mask = false_negative_mask(data.train_pairs, data.negative_exclusions).to(device)
    val_mask = false_negative_mask(
        data.validation_pairs, data.negative_exclusions
    ).to(device)

    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "config_snapshot.yaml").write_text(config_snapshot_text, encoding="utf-8")
    checkpoint_path = seed_dir / "best_checkpoint.pt"
    checkpoint_manifest_path = seed_dir / "checkpoint_manifest.json"
    epoch_rows: list[dict[str, Any]] = []
    best_value = math.inf
    best_epoch = 0
    best_mrr = 0.0
    best_manifest: dict[str, Any] | None = None
    no_improve = 0
    final_epoch = 0
    total_steps = 0
    max_epochs = 1 if smoke_test else config.max_epochs
    _progress(
        progress_callback,
        event="seed_start",
        stage="clip_training",
        direction=direction,
        seed=seed,
        max_epochs=max_epochs,
        elapsed_seconds=0.0,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(seed * 100_000 + epoch)
        order = torch.randperm(train_text.shape[0], generator=generator).tolist()
        batch_losses = []
        grad_norms = []
        masked_counts = []
        batch_count = math.ceil(len(order) / config.batch_size)
        for batch_index, start in enumerate(
            range(0, len(order), config.batch_size), start=1
        ):
            indexes = order[start : start + config.batch_size]
            if len(indexes) < 2:
                continue
            idx = torch.tensor(indexes, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            text_proj, stat_proj = model(train_text[idx], train_stat[idx])
            batch_mask = train_mask[idx][:, idx]
            output = symmetric_masked_contrastive_loss(
                text_proj,
                stat_proj,
                temperature=model.temperature(),
                false_negative_mask=batch_mask,
            )
            output.loss.backward()
            grad_norm = _gradient_norm(model)
            if config.gradient_clipping_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(output.loss.detach().cpu().item()))
            grad_norms.append(float(grad_norm))
            masked_counts.append(int(output.masked_negative_count))
            total_steps += 1
            if (
                batch_index == 1
                or batch_index % batch_log_interval == 0
                or batch_index == batch_count
            ):
                _progress(
                    progress_callback,
                    event="train_batch",
                    stage="clip_training",
                    direction=direction,
                    seed=seed,
                    epoch=epoch,
                    batch=batch_index,
                    batch_count=batch_count,
                    metrics={"loss": batch_losses[-1]},
                    elapsed_seconds=time.perf_counter() - started,
                )
            if smoke_test and total_steps >= config.smoke_test_steps:
                break
        final_epoch = epoch
        train_metrics, val_metrics, rep_metrics = evaluate_model(
            model=model,
            train_text=train_text,
            train_stat=train_stat,
            train_mask=train_mask,
            val_text=val_text,
            val_stat=val_stat,
            val_mask=val_mask,
            thresholds=config.collapse_thresholds,
        )
        val_loss = float(val_metrics["validation_contrastive_loss"])
        row = {
            "seed": int(seed),
            "epoch": int(epoch),
            "training_loss": float(np.mean(batch_losses)) if batch_losses else math.nan,
            "validation_loss": val_loss,
            "validation_mrr": float(val_metrics["mean_reciprocal_rank"]),
            "learning_rate": float(config.learning_rate),
            "gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "masked_negative_count": int(np.sum(masked_counts)) if masked_counts else 0,
            "temperature": float(model.temperature().detach().cpu().item()),
            "train_positive_margin": train_metrics["positive_minus_negative_margin"],
            "validation_positive_margin": val_metrics["positive_minus_negative_margin"],
        }
        for key, value in train_metrics.items():
            if key != "split":
                row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            if key != "split":
                row[f"validation_{key}"] = value
        epoch_rows.append(row)
        improved = val_loss < best_value - config.minimum_improvement
        if improved:
            best_value = val_loss
            best_epoch = epoch
            best_mrr = float(val_metrics["mean_reciprocal_rank"])
            no_improve = 0
            best_manifest = save_checkpoint(
                model=model,
                path=checkpoint_path,
                manifest_path=checkpoint_manifest_path,
                seed=seed,
                epoch=epoch,
                validation_metric=config.selection_metric,
                validation_value=best_value,
                parameter_count=parameter_count,
                upstream_hashes=data.upstream_hashes,
                git_commit=_git_commit(),
                statistical_view_scope=config.statistical_view_scope,
                extra={
                    "source_dataset": data.training_dataset,
                    "external_dataset": data.external_dataset,
                    "fit_scope": f"{data.training_dataset}_contrastive_training_features_dev_only",
                    "pairing_policy_version": "identity_equivalence_v2",
                    "configuration_hash": config.configuration_hash,
                    "data_manifest_hash": config.data_manifest_hash,
                    "statistical_preprocessor_hash": config.statistical_preprocessor_hash,
                    "source_anchor_hash": config.source_anchor_hash,
                    "initial_temperature": float(config.model.initial_temperature),
                    "final_temperature": float(model.temperature().detach().cpu().item()),
                    "statistical_fields": data.statistical_fields,
                    "statistical_view_limitation": _statistical_limitation(config, data),
                },
            )
            _progress(
                progress_callback,
                event="new_best_checkpoint",
                stage="clip_training",
                direction=direction,
                seed=seed,
                epoch=epoch,
                metrics={
                    "validation_loss": best_value,
                    "mrr": best_mrr,
                    "checkpoint_sha256": best_manifest["checkpoint_sha256"],
                },
                elapsed_seconds=time.perf_counter() - started,
            )
        else:
            no_improve += 1
        _progress(
            progress_callback,
            event="epoch_end",
            stage="clip_training",
            direction=direction,
            seed=seed,
            epoch=epoch,
            metrics={
                "train_loss": row["training_loss"],
                "validation_loss": val_loss,
                "mrr": float(val_metrics["mean_reciprocal_rank"]),
                "recall_at_1": float(
                    (
                        val_metrics["text_to_statistical_recall_at_1"]
                        + val_metrics["statistical_to_text_recall_at_1"]
                    )
                    / 2.0
                ),
                "recall_at_5": float(
                    (
                        val_metrics["text_to_statistical_recall_at_5"]
                        + val_metrics["statistical_to_text_recall_at_5"]
                    )
                    / 2.0
                ),
                "recall_at_10": float(
                    (
                        val_metrics["text_to_statistical_recall_at_10"]
                        + val_metrics["statistical_to_text_recall_at_10"]
                    )
                    / 2.0
                ),
                "best_validation_loss": best_value,
                "patience": no_improve,
                "patience_limit": config.early_stopping_patience,
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        if smoke_test and total_steps >= config.smoke_test_steps:
            break
        if no_improve >= config.early_stopping_patience:
            _progress(
                progress_callback,
                event="early_stop",
                stage="clip_training",
                direction=direction,
                seed=seed,
                epoch=epoch,
                metrics={"best_validation_loss": best_value},
                elapsed_seconds=time.perf_counter() - started,
            )
            break

    if best_manifest is None:
        raise RuntimeError(f"seed {seed}: no checkpoint was created")
    epoch_frame = pd.DataFrame(epoch_rows)
    epoch_metrics_path = seed_dir / "epoch_metrics.csv"
    epoch_frame.to_csv(epoch_metrics_path, index=False)
    representation_metrics_path = seed_dir / "representation_metrics.json"
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    best_train_metrics, best_val_metrics, rep_metrics = evaluate_model(
        model=model,
        train_text=train_text,
        train_stat=train_stat,
        train_mask=train_mask,
        val_text=val_text,
        val_stat=val_stat,
        val_mask=val_mask,
        thresholds=config.collapse_thresholds,
    )
    write_json(
        representation_metrics_path,
        {
            **rep_metrics,
            "checkpoint_epoch": int(best_epoch),
            "checkpoint_sha256": str(best_manifest["checkpoint_sha256"]),
            "train_retrieval": best_train_metrics,
            "validation_retrieval": best_val_metrics,
            "collapse_diagnostics": rep_metrics,
        },
    )
    training_log_path = seed_dir / "training_log.json"
    write_json(
        training_log_path,
        {
            "seed": int(seed),
            "initial_random_state": {"python_random_seed": int(seed), "numpy_seed": int(seed), "torch_seed": int(seed)},
            "best_epoch": int(best_epoch),
            "final_epoch": int(final_epoch),
            "early_stopping_epoch": int(final_epoch),
            "total_optimizer_steps": int(total_steps),
            "model_trained": True,
            "source_dataset": data.training_dataset,
            "external_dataset": data.external_dataset,
            "external_dataset_used_for_training": False,
            "external_dataset_used_for_model_selection": False,
            "statistical_view_scope": config.statistical_view_scope,
        },
    )
    _progress(
        progress_callback,
        event="seed_end",
        stage="clip_training",
        direction=direction,
        seed=seed,
        epoch=final_epoch,
        metrics={
            "best_epoch": best_epoch,
            "stop_epoch": final_epoch,
            "best_validation_loss": best_value,
            "mrr": float(best_val_metrics["mean_reciprocal_rank"]),
            "recall_at_1": float(
                (
                    best_val_metrics["text_to_statistical_recall_at_1"]
                    + best_val_metrics["statistical_to_text_recall_at_1"]
                )
                / 2.0
            ),
            "recall_at_5": float(
                (
                    best_val_metrics["text_to_statistical_recall_at_5"]
                    + best_val_metrics["statistical_to_text_recall_at_5"]
                )
                / 2.0
            ),
            "recall_at_10": float(
                (
                    best_val_metrics["text_to_statistical_recall_at_10"]
                    + best_val_metrics["statistical_to_text_recall_at_10"]
                )
                / 2.0
            ),
            "checkpoint_sha256": str(best_manifest["checkpoint_sha256"]),
        },
        elapsed_seconds=time.perf_counter() - started,
    )
    return SeedTrainingResult(
        seed=seed,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        early_stopping_epoch=final_epoch,
        best_validation_loss=best_value,
        best_validation_mrr=best_mrr,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_hash=str(best_manifest["checkpoint_sha256"]),
        parameter_count=parameter_count,
        epoch_metrics_path=epoch_metrics_path,
        representation_metrics_path=representation_metrics_path,
        training_log_path=training_log_path,
    )


def evaluate_model(
    *,
    model: SemanticStatisticalContrastiveEncoder,
    train_text: torch.Tensor,
    train_stat: torch.Tensor,
    train_mask: torch.Tensor,
    val_text: torch.Tensor,
    val_stat: torch.Tensor,
    val_mask: torch.Tensor,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        train_text_proj, train_stat_proj = model(train_text, train_stat)
        val_text_proj, val_stat_proj = model(val_text, val_stat)
        train_metrics = retrieval_metrics(
            train_text_proj,
            train_stat_proj,
            false_negative_mask=train_mask,
            temperature=model.temperature(),
            split="train",
        )
        val_metrics = retrieval_metrics(
            val_text_proj,
            val_stat_proj,
            false_negative_mask=val_mask,
            temperature=model.temperature(),
            split="validation",
        )
        representation = {
            "train_text": collapse_diagnostics(train_text_proj, thresholds=thresholds, label="train_projected_text"),
            "train_statistical": collapse_diagnostics(train_stat_proj, thresholds=thresholds, label="train_projected_statistical"),
            "validation_text": collapse_diagnostics(val_text_proj, thresholds=thresholds, label="validation_projected_text"),
            "validation_statistical": collapse_diagnostics(val_stat_proj, thresholds=thresholds, label="validation_projected_statistical"),
        }
    return train_metrics, val_metrics, representation


def _set_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).cpu().item())
        total += value * value
    return float(total**0.5)


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _statistical_limitation(config: ClipTrainingConfig, data: TrainingDataBundle) -> str:
    if data.statistical_dim == 1 and config.statistical_view_scope == "missingness_only":
        return "architectural proof of concept: aligns feature semantics primarily with DEV missingness behavior"
    return "approved multi-dimensional statistical view"


def _progress(
    callback: Callable[[dict[str, Any]], None] | None,
    **event: Any,
) -> None:
    if callback is not None:
        callback(event)
