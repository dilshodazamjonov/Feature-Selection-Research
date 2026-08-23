"""Frozen CLIP experiment for Home Credit Model Stability 2024.

This module is deliberately isolated from the existing locked Stability results.
It authenticates Prompt-1 and corrected historical source artifacts, trains only
the new Stability representation, produces three target-free rankings, and then
runs the six frozen downstream cells behind an explicit pre-OOT integrity gate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk_fs.clip.model import (
    ClipModelConfig,
    SemanticStatisticalContrastiveEncoder,
    count_trainable_parameters,
)
from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.source_anchor import vector_hash
from credit_risk_fs.clip.trainer import SeedTrainingResult, train_seed
from credit_risk_fs.clip.training_validation import (
    ClipTrainingConfig,
    TrainingDataBundle,
    tensors_for_pairs,
)
from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.evaluation.drift import calculate_psi, jaccard_similarity
from credit_risk_fs.experiments.prompt_16_third_dataset import (
    _matrix_identity,
    _read_date_slice,
    _validate_scope_frame,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder, SparsePreprocessor
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector


REQUIRED_SEEDS = (11, 22, 33, 44, 55)
REFERENCE_SEED = 11
FEATURE_COUNT = 1959
JOINT_DIMENSION = 32
NON_PREDICTORS = ("case_id", "date_decision", "target")
DESCRIPTOR_FIELDS = (
    "missing_rate",
    "unique_ratio",
    "concentration_share",
    "signed_log_mean",
    "log_standard_deviation",
    "clipped_skewness",
    "normalized_entropy",
    "is_numeric",
    "is_categorical",
    "is_binary",
    "numeric_stats_valid",
    "skewness_valid",
    "entropy_valid",
)
FORBIDDEN_CLIP_TOKENS = (
    "target",
    "date_decision",
    "case_id",
    "oot",
    "auc",
    "prediction",
    "model_output",
    "selector_rank",
    "llm_rank",
    "rfe_rank",
    "boruta",
    "shap",
)


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if str(column).startswith("embedding_") and str(column)[10:].isdigit()
    )


class ExperimentContractError(RuntimeError):
    """Raised when a frozen scientific or integrity contract is violated."""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    _atomic_bytes(output, (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"))
    return output


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(path)
    _atomic_bytes(output, frame.to_csv(index=False).encode("utf-8"))
    return output


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentContractError(f"JSON contract is not an object: {path}")
    return value


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ExperimentContractError(f"{label} mismatch: expected={expected!r}, observed={observed!r}")


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ExperimentContractError(f"missing {label}: {path}")
    observed = sha256_file(path)
    _require_equal(observed, expected, f"{label} SHA-256")
    return observed


def _resolve(root: Path, configured: str | Path) -> Path:
    candidate = Path(configured)
    return candidate if candidate.is_absolute() else root / candidate


class ProgressLogger:
    """Mirrors progress to stdout, a durable text log, and schema-stable JSONL."""

    required_fields = ("timestamp", "stage", "event", "elapsed_seconds")

    def __init__(self, artifact_root: str | Path, run_log: str, progress_jsonl: str) -> None:
        self.root = Path(artifact_root)
        self.run_log = self.root / run_log
        self.progress_jsonl = self.root / progress_jsonl
        self.run_log.parent.mkdir(parents=True, exist_ok=True)
        self.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._lock = threading.Lock()

    def emit(
        self,
        message: str,
        *,
        stage: str,
        event: str,
        direction: str | None = None,
        seed: int | None = None,
        epoch: int | None = None,
        batch: int | None = None,
        fold: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage),
            "direction": direction,
            "seed": seed,
            "epoch": epoch,
            "batch": batch,
            "fold": fold,
            "event": str(event),
            "metrics": dict(metrics or {}),
            "elapsed_seconds": float(
                time.perf_counter() - self.started if elapsed_seconds is None else elapsed_seconds
            ),
        }
        serialized = json.dumps(row, sort_keys=True, default=str)
        line = f"{row['timestamp']} {message}"
        with self._lock:
            print(message, flush=True)
            with self.run_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
            with self.progress_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
        return row

    def callback(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event", "progress"))
        seed = payload.get("seed")
        epoch = payload.get("epoch")
        batch = payload.get("batch")
        direction = payload.get("direction")
        metrics = dict(payload.get("metrics", {}))
        if event == "seed_start":
            message = f"[CLIP][Stability][seed {seed}] starting"
        elif event == "train_batch":
            message = f"[CLIP][Stability][seed {seed}][epoch {int(epoch):02d}][train batch {batch}/{payload.get('batch_count')}]"
        elif event == "epoch_end":
            message = f"[CLIP][Stability][seed {seed}][epoch {int(epoch):02d}] " + " ".join(
                f"{key}={value}" for key, value in metrics.items()
            )
        elif event == "new_best_checkpoint":
            message = f"[CLIP][Stability][seed {seed}][epoch {int(epoch):02d}] new best checkpoint"
        elif event == "early_stop":
            message = f"[CLIP][Stability][seed {seed}] early stop at epoch {epoch}"
        elif event == "seed_end":
            message = f"[CLIP][Stability][seed {seed}] complete checkpoint_sha256={metrics.get('checkpoint_sha256')}"
        else:
            message = f"[CLIP][Stability] {event}"
        self.emit(
            message,
            stage=str(payload.get("stage", "clip_training")),
            event=event,
            direction=direction,
            seed=seed,
            epoch=epoch,
            batch=batch,
            metrics=metrics,
            elapsed_seconds=payload.get("elapsed_seconds"),
        )

    @contextmanager
    def heartbeat(self, message: str, *, stage: str, interval: float = 30.0, **context: Any):
        stop = threading.Event()

        def worker() -> None:
            while not stop.wait(interval):
                self.emit(message, stage=stage, event="heartbeat", **context)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=max(1.0, interval))


class StageStore:
    """Hash-authenticated COMPLETE markers for safe, fail-closed resume."""

    def __init__(self, root: str | Path, config_hash: str) -> None:
        self.root = Path(root) / "manifests" / "stages"
        self.config_hash = config_hash

    def path(self, stage: str) -> Path:
        return self.root / f"{stage}.json"

    def reusable(self, stage: str, input_hashes: Mapping[str, str]) -> bool:
        path = self.path(stage)
        if not path.exists():
            return False
        manifest = _read_json(path)
        _require_equal(manifest.get("status"), "COMPLETE", f"stage {stage} status")
        _require_equal(manifest.get("config_hash"), self.config_hash, f"stage {stage} config hash")
        _require_equal(manifest.get("input_hashes"), dict(input_hashes), f"stage {stage} input hashes")
        for raw_path, expected in manifest.get("output_hashes", {}).items():
            _require_hash(Path(raw_path), str(expected), f"stage {stage} output")
        return True

    def complete(
        self,
        stage: str,
        input_hashes: Mapping[str, str],
        outputs: Sequence[str | Path],
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        output_hashes = {}
        for raw in outputs:
            path = Path(raw).resolve()
            if not path.is_file():
                raise ExperimentContractError(f"stage {stage} cannot complete; output missing: {path}")
            output_hashes[path.as_posix()] = sha256_file(path)
        return atomic_json(
            self.path(stage),
            {
                "schema_version": "clip_experiment_stage_v1",
                "status": "COMPLETE",
                "stage": stage,
                "config_hash": self.config_hash,
                "input_hashes": dict(input_hashes),
                "output_hashes": output_hashes,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                **dict(extra or {}),
            },
        )

    def archive_incomplete_seed(self, seed_dir: str | Path) -> Path | None:
        path = Path(seed_dir)
        if not path.exists():
            return None
        marker = path / "stage_complete.json"
        if marker.exists():
            return None
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = path.with_name(f"{path.name}.incomplete.{suffix}")
        collision = 1
        while destination.exists():
            destination = path.with_name(
                f"{path.name}.incomplete.{suffix}.{collision}"
            )
            collision += 1
        path.rename(destination)
        return destination


@dataclass(frozen=True)
class Prompt1Package:
    root: Path
    manifest_sha256: str
    feature_universe_hash: str
    feature_universe: pd.DataFrame
    pair_frame: pd.DataFrame
    raw_descriptors: pd.DataFrame
    anchor_feature_ids: tuple[str, ...]
    statistical_preprocessor_identity: str
    text_embedding_identity: str


class Prompt1PackageValidator:
    """Authenticates Prompt-1 without rebuilding or mutating any artifact."""

    def __init__(self, repository_root: str | Path, config: Mapping[str, Any]) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.config = dict(config)

    def validate(self) -> Prompt1Package:
        root = _resolve(self.repository_root, self.config["path"]).resolve()
        if not root.is_dir():
            raise ExperimentContractError(f"Prompt-1 package is missing: {root}")
        manifest_path = root / self.config["sha256_manifest_path"]
        manifest_hash = _require_hash(
            manifest_path,
            self.config["sha256_manifest_sha256"],
            "Prompt-1 SHA manifest",
        )
        manifest = pd.read_csv(manifest_path)
        required_columns = {"relative_path", "sha256", "size_bytes"}
        if not required_columns.issubset(manifest.columns):
            raise ExperimentContractError("Prompt-1 SHA manifest schema changed")
        indexed = set(manifest["relative_path"].astype(str).str.replace("\\", "/"))
        for row in manifest.to_dict("records"):
            relative = str(row["relative_path"]).replace("\\", "/")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ExperimentContractError(f"Prompt-1 SHA manifest path escapes package: {relative}")
            _require_hash(path, str(row["sha256"]), f"Prompt-1 {relative}")
            _require_equal(path.stat().st_size, int(row["size_bytes"]), f"Prompt-1 {relative} size")
        for relative in self.config["required_artifacts"]:
            if relative == self.config["sha256_manifest_path"]:
                if not manifest_path.is_file():
                    raise ExperimentContractError("Prompt-1 self manifest missing")
                continue
            if relative not in indexed:
                raise ExperimentContractError(f"Prompt-1 artifact absent from SHA manifest: {relative}")

        validation = _read_json(root / self.config["validation_report_path"])
        _require_equal(validation.get("overall_status"), self.config["required_status"], "Prompt-1 validation")
        if any(item.get("status") != "PASS" for item in validation.get("checks", [])):
            raise ExperimentContractError("Prompt-1 contains a non-PASS validation check")
        methodology = _read_json(root / "methodology_lock.json")
        _require_equal(methodology.get("negative_policy"), "identity_equivalence_v2", "Prompt-1 pairing policy")
        _require_equal(methodology.get("feature_universe"), FEATURE_COUNT, "Prompt-1 feature count")
        _require_equal(methodology.get("feature_universe_hash"), self.config["feature_universe_hash"], "Prompt-1 feature hash")
        _require_equal(methodology.get("planned_directions"), [
            "Stability->Stability", "HomeCredit->Stability", "LendingClub->Stability"
        ], "Prompt-1 direction lock")

        universe = pd.read_csv(root / "metadata/feature_universe.csv")
        if len(universe) != FEATURE_COUNT or not universe["feature_name"].is_unique:
            raise ExperimentContractError("Prompt-1 feature universe is not exactly 1,959 unique names")
        forbidden = set(name.casefold() for name in NON_PREDICTORS) & set(
            universe["feature_name"].astype(str).str.casefold()
        )
        if forbidden:
            raise ExperimentContractError(f"non-predictors entered Prompt-1 feature identities: {sorted(forbidden)}")
        if not universe["feature_id"].astype(str).is_unique:
            raise ExperimentContractError("Prompt-1 feature IDs are not unique")

        text_manifest = _read_json(root / "text/text_embedding_manifest.json")
        expected_text = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": "main",
            "template_version": "feature_text_v1",
            "embedding_dimension": 384,
            "normalization": "L2",
            "frozen_inference_only": True,
            "fine_tuned": False,
            "row_count": FEATURE_COUNT,
        }
        for key, expected in expected_text.items():
            _require_equal(text_manifest.get(key), expected, f"Prompt-1 text {key}")

        preprocessor = _read_json(root / "statistics/stability_source_stat_preprocessor.json")
        _require_equal(preprocessor.get("descriptor_order"), list(DESCRIPTOR_FIELDS), "Prompt-1 descriptor order")
        _require_equal(preprocessor.get("fit_feature_count"), 1564, "Prompt-1 preprocessor fit count")
        _require_equal(preprocessor.get("fit_split"), "representation_train_feature_identities_only", "Prompt-1 preprocessor fit scope")
        _require_equal(preprocessor.get("validation_feature_identities_used_for_fit"), False, "Prompt-1 validation fit use")
        _require_equal(preprocessor.get("target_used"), False, "Prompt-1 target use")
        _require_equal(preprocessor.get("oot_used"), False, "Prompt-1 OOT use")
        _require_equal([preprocessor.get("clip_min"), preprocessor.get("clip_max")], [-8.0, 8.0], "Prompt-1 clipping")

        pair_manifest = _read_json(root / "pairs/stability_source_pairs_manifest.json")
        for key in ("target_included", "oot_values_included", "selector_or_model_outputs_included"):
            _require_equal(pair_manifest.get(key), False, f"Prompt-1 pair boundary {key}")
        _require_equal(pair_manifest.get("row_count"), FEATURE_COUNT, "Prompt-1 pair rows")
        _require_equal(pair_manifest.get("text_embedding_dimension"), 384, "Prompt-1 pair text dimension")
        _require_equal(pair_manifest.get("statistical_dimension"), 13, "Prompt-1 pair stat dimension")
        pairs = pd.read_parquet(root / "pairs/stability_source_pairs.parquet")
        forbidden_columns = [
            column for column in pairs.columns
            if any(token in str(column).casefold() for token in FORBIDDEN_CLIP_TOKENS)
        ]
        if forbidden_columns:
            raise ExperimentContractError(f"forbidden CLIP fields in Prompt-1 pairs: {forbidden_columns}")
        if len(pairs) != FEATURE_COUNT or not pairs["feature_id"].is_unique:
            raise ExperimentContractError("Prompt-1 pair identities are not exactly one per feature")
        text_columns = _embedding_columns(pairs)
        stat_columns = [f"stat_{name}" for name in DESCRIPTOR_FIELDS]
        _require_equal(len(text_columns), 384, "Prompt-1 pair text width")
        if any(column not in pairs for column in stat_columns):
            raise ExperimentContractError("Prompt-1 pair statistical schema is incomplete")
        values = pairs[[*text_columns, *stat_columns]].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ExperimentContractError("Prompt-1 pair vectors contain NaN/Inf")

        raw = pd.read_parquet(root / "statistics/statistical_descriptors_raw.parquet")
        descriptor_positions = [raw.columns.get_loc(field) for field in DESCRIPTOR_FIELDS if field in raw.columns]
        if len(descriptor_positions) != 13 or descriptor_positions != list(range(descriptor_positions[0], descriptor_positions[0] + 13)):
            raise ExperimentContractError("Prompt-1 raw descriptor order changed")
        if len(raw) != FEATURE_COUNT or not raw["feature_id"].is_unique:
            raise ExperimentContractError("Prompt-1 raw descriptor identities changed")
        if not np.isfinite(raw[list(DESCRIPTOR_FIELDS)].to_numpy(dtype=np.float64)).all():
            raise ExperimentContractError("Prompt-1 raw descriptors contain NaN/Inf")

        anchor_manifest = _read_json(root / "anchor/stability_source_anchor_manifest.json")
        _require_equal(anchor_manifest.get("actual_member_count"), 23, "Prompt-1 anchor size")
        _require_equal(anchor_manifest.get("target_used"), False, "Prompt-1 anchor target use")
        _require_equal(anchor_manifest.get("oot_used"), False, "Prompt-1 anchor OOT use")
        anchor_ids = tuple(str(value) for value in anchor_manifest["anchor_feature_ids"])
        if len(anchor_ids) != 23 or not set(anchor_ids).issubset(set(pairs["feature_id"].astype(str))):
            raise ExperimentContractError("Prompt-1 anchor identities do not reconcile")
        return Prompt1Package(
            root=root,
            manifest_sha256=manifest_hash,
            feature_universe_hash=str(self.config["feature_universe_hash"]),
            feature_universe=universe,
            pair_frame=pairs,
            raw_descriptors=raw,
            anchor_feature_ids=anchor_ids,
            statistical_preprocessor_identity=str(preprocessor["internal_preprocessor_hash"]),
            text_embedding_identity=str(text_manifest["cache_identity"]),
        )


@dataclass(frozen=True)
class AuthenticatedSource:
    name: str
    dataset: str
    root: Path
    checkpoint_paths: Mapping[int, Path]
    checkpoint_hashes: Mapping[int, str]
    preprocessor: Any
    preprocessor_identity: str
    anchor_vectors: Mapping[int, np.ndarray]
    anchor_identity: str
    authentication: Mapping[str, Any]


class FrozenTransform:
    """Read-only transform facade: intentionally exposes no fit method."""

    def __init__(self, transform: Callable[[Any], Any], identity: str) -> None:
        self._transform = transform
        self.identity = identity

    def transform(self, values: Any) -> np.ndarray:
        output = np.asarray(self._transform(values), dtype=np.float32)
        if output.ndim != 2 or output.shape[1] != len(DESCRIPTOR_FIELDS):
            raise ExperimentContractError("source preprocessor returned the wrong shape")
        if not np.isfinite(output).all():
            raise ExperimentContractError("source preprocessor returned NaN/Inf")
        return output


class HistoricalSourceArtifactResolver:
    """Authenticates explicit corrected HC/LC roots without directory scanning."""

    def __init__(self, representation_contract: Mapping[str, Any], seeds: Sequence[int]) -> None:
        self.contract = dict(representation_contract)
        _require_equal(tuple(seeds), REQUIRED_SEEDS, "historical source seed set")
        self.seeds = tuple(seeds)
        self.model_config = self._model_config(self.contract)

    @staticmethod
    def _model_config(contract: Mapping[str, Any]) -> ClipModelConfig:
        architecture = dict(contract["architecture"])
        architecture.pop("expected_parameter_count", None)
        return ClipModelConfig(**architecture)

    def authenticate(self, name: str, config: Mapping[str, Any]) -> AuthenticatedSource:
        if name not in {"homecredit", "lendingclub"}:
            raise ExperimentContractError(f"unsupported historical source: {name}")
        root = Path(config["root"]).resolve()
        if not root.is_dir():
            raise ExperimentContractError(f"corrected {name} source root missing: {root}")
        evidence_files = (
            ["training_manifest", "negative_policy_manifest", "tensor_schema"]
            if name == "homecredit"
            else ["training_stage_manifest", "projection_manifest", "data_manifest", "negative_policy_manifest"]
        )
        evidence: dict[str, Any] = {}
        for key in evidence_files:
            path = _resolve(root, config[key])
            evidence[key] = {"path": path.as_posix(), "sha256": sha256_file(path), "payload": _read_json(path)}
        joined = json.dumps([item["payload"] for item in evidence.values()], sort_keys=True).casefold()
        if "identity_equivalence_v2" not in joined:
            raise ExperimentContractError(f"{name} corrected identity_equivalence_v2 evidence is absent")
        dataset = str(config["dataset"])
        if dataset.casefold() not in joined:
            raise ExperimentContractError(f"{name} source dataset identity is absent from manifests")

        checkpoint_paths: dict[int, Path] = {}
        checkpoint_hashes: dict[int, str] = {}
        model_config = self._model_config(self.contract)
        for seed in self.seeds:
            path = root / str(config["checkpoint_path_template"]).format(seed=seed)
            expected = str(config["checkpoint_hashes"][str(seed)])
            checkpoint_hashes[seed] = _require_hash(path, expected, f"{name} seed {seed} checkpoint")
            manifest_path = root / str(config["checkpoint_manifest_template"]).format(seed=seed)
            manifest = _read_json(manifest_path)
            _require_equal(int(manifest.get("seed")), seed, f"{name} checkpoint seed")
            _require_equal(manifest.get("checkpoint_sha256"), expected, f"{name} checkpoint manifest hash")
            _require_equal(int(manifest.get("parameter_count")), 27488, f"{name} parameter count")
            checkpoint_paths[seed] = path
            model = SemanticStatisticalContrastiveEncoder(model_config)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
            expected_model_payload = asdict(model_config)
            if "model_config" in payload:
                _require_equal(payload["model_config"], expected_model_payload, f"{name} seed {seed} model config")
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ExperimentContractError(f"{name} seed {seed} architecture incompatible: {exc}") from exc
            _require_equal(count_trainable_parameters(model), 27488, f"{name} active architecture")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise ExperimentContractError(f"{name} source projector was not frozen")

        text_manifest_path = Path(config["text_manifest_path"])
        _require_hash(text_manifest_path, str(config["text_manifest_sha256"]), f"{name} text manifest")
        text_manifest = _read_json(text_manifest_path)
        text_text = json.dumps(text_manifest, sort_keys=True)
        for expected in ("sentence-transformers/all-MiniLM-L6-v2", "feature_text_v1", "384"):
            if expected not in text_text:
                raise ExperimentContractError(f"{name} text contract lacks {expected}")

        if name == "homecredit":
            preprocessor_path = Path(config["preprocessor_path"])
            _require_hash(preprocessor_path, str(config["preprocessor_file_sha256"]), "HC preprocessor")
            payload = _read_json(preprocessor_path)
            identity = str(payload.get("preprocessor_hash", payload.get("internal_preprocessor_hash")))
            _require_equal(identity, config["preprocessor_identity"], "HC preprocessor identity")
            _require_equal(payload.get("field_order"), list(DESCRIPTOR_FIELDS), "HC descriptor field order")
            _require_equal(payload.get("fit_dataset"), "homecredit", "HC preprocessor fit dataset")
            _require_equal(payload.get("fit_split"), "train", "HC preprocessor fit split")
            medians = np.array([payload["medians"][field] for field in DESCRIPTOR_FIELDS[:7]], dtype=np.float64)
            iqrs = np.array([payload["iqr"][field] for field in DESCRIPTOR_FIELDS[:7]], dtype=np.float64)
            iqrs = np.where(iqrs == 0.0, 1.0, iqrs)

            def hc_transform(values: Any) -> np.ndarray:
                array = _descriptor_array(values)
                output = array.copy()
                output[:, :7] = np.clip((output[:, :7] - medians) / iqrs, -8.0, 8.0)
                return output

            preprocessor = FrozenTransform(hc_transform, identity)
            anchor_path = root / config["anchor_path"]
            _require_hash(anchor_path, str(config["anchor_file_sha256"]), "HC source anchor")
            anchor_manifest_path = root / config["anchor_manifest"]
            _require_hash(anchor_manifest_path, str(config["anchor_manifest_file_sha256"]), "HC anchor manifest")
            anchor_manifest = _read_json(anchor_manifest_path)
            _require_equal(int(anchor_manifest.get("anchor_count", anchor_manifest.get("actual_member_count", anchor_manifest.get("member_count", -1)))), 23, "HC anchor member count")
            _require_equal(anchor_manifest.get("target_used"), False, "HC anchor target use")
            _require_equal(anchor_manifest.get("oot_used"), False, "HC anchor OOT use")
            _require_equal(anchor_manifest.get("pairing_policy_version"), "identity_equivalence_v2", "HC anchor pairing policy")
            _require_equal(anchor_manifest.get("anchor_hash"), config["anchor_identity"], "HC anchor identity")
            anchor = _normalized_vector(np.load(anchor_path), "HC anchor")
            anchor_vectors = {seed: anchor.copy() for seed in self.seeds}
            anchor_identity = str(config["anchor_identity"])
        else:
            preprocessor_path = root / config["preprocessor_path"]
            _require_hash(preprocessor_path, str(config["preprocessor_file_sha256"]), "LC preprocessor")
            preprocessor_json = root / config["preprocessor_json"]
            _require_hash(preprocessor_json, str(config["preprocessor_json_sha256"]), "LC preprocessor JSON")
            preprocessor_payload = _read_json(preprocessor_json)
            identity = str(preprocessor_payload.get("preprocessor_hash", preprocessor_payload.get("internal_preprocessor_hash")))
            _require_equal(identity, config["preprocessor_identity"], "LC preprocessor identity")
            _require_equal(preprocessor_payload.get("field_order"), list(DESCRIPTOR_FIELDS), "LC descriptor field order")
            _require_equal(preprocessor_payload.get("fit_dataset"), "lendingclub_v2", "LC preprocessor fit dataset")
            _require_equal(preprocessor_payload.get("fit_split"), "train", "LC preprocessor fit split")
            _require_equal(preprocessor_payload.get("clipping_enabled"), False, "LC clipping policy")
            fitted = joblib.load(preprocessor_path)
            if not callable(getattr(fitted, "transform", None)):
                raise ExperimentContractError("LC authenticated preprocessor has no transform")
            preprocessor = FrozenTransform(lambda values: fitted.transform(_descriptor_frame(values)), identity)
            anchor_manifest_path = root / config["source_anchor_manifest"]
            _require_hash(anchor_manifest_path, str(config["source_anchor_manifest_sha256"]), "LC anchor manifest")
            anchor_manifest = _read_json(anchor_manifest_path)
            _require_equal(anchor_manifest.get("source_dataset"), "lendingclub_v2", "LC anchor source dataset")
            _require_equal(anchor_manifest.get("actual_member_count"), 23, "LC anchor member count")
            _require_equal(anchor_manifest.get("target_used"), False, "LC anchor target use")
            _require_equal(anchor_manifest.get("oot_used"), False, "LC anchor OOT use")
            _require_equal(anchor_manifest.get("external_data_used"), False, "LC anchor external use")
            _require_equal(anchor_manifest.get("pairing_policy_version"), "identity_equivalence_v2", "LC anchor pairing policy")
            _require_equal(anchor_manifest.get("anchor_hashes_by_seed"), config["anchor_hashes"], "LC seed anchor identities")
            _require_equal(anchor_manifest.get("checkpoint_hashes_by_seed"), config["checkpoint_hashes"], "LC anchor checkpoint identities")
            anchor_vectors = {}
            for seed in self.seeds:
                anchor_path = root / str(config["seed_anchor_path_template"]).format(seed=seed)
                anchor = _normalized_vector(np.load(anchor_path), f"LC seed {seed} anchor")
                expected_anchor = str(config["anchor_hashes"][str(seed)])
                manifest_path = root / str(config["seed_anchor_manifest_template"]).format(seed=seed)
                manifest = _read_json(manifest_path)
                observed_identity = str(manifest.get("anchor_sha256", manifest.get("anchor_hash", "")))
                _require_equal(observed_identity, expected_anchor, f"LC seed {seed} anchor manifest identity")
                _require_equal(vector_hash(anchor), expected_anchor, f"LC seed {seed} anchor vector identity")
                _require_equal(manifest.get("checkpoint_hash"), checkpoint_hashes[seed], f"LC seed {seed} anchor checkpoint")
                _require_equal(manifest.get("target_used"), False, f"LC seed {seed} anchor target use")
                _require_equal(manifest.get("oot_used"), False, f"LC seed {seed} anchor OOT use")
                anchor_vectors[seed] = anchor
            anchor_identity = canonical_hash({str(seed): config["anchor_hashes"][str(seed)] for seed in self.seeds})

        authentication = {
            "schema_version": "corrected_source_authentication_v1",
            "status": "PASS",
            "source": name,
            "dataset": dataset,
            "generation": config["generation"],
            "root": root.as_posix(),
            "root_read_only_contract": True,
            "seeds": list(self.seeds),
            "checkpoint_hashes": {str(key): value for key, value in checkpoint_hashes.items()},
            "architecture": asdict(model_config),
            "parameter_count": 27488,
            "pairing_policy": "identity_equivalence_v2",
            "preprocessor_identity": identity,
            "preprocessor_policy": config["preprocessor_policy"],
            "anchor_identity": anchor_identity,
            "anchor_member_count": int(config["anchor_member_count"]),
            "evidence": {key: {k: v for k, v in value.items() if k != "payload"} for key, value in evidence.items()},
        }
        return AuthenticatedSource(
            name=name,
            dataset=dataset,
            root=root,
            checkpoint_paths=checkpoint_paths,
            checkpoint_hashes=checkpoint_hashes,
            preprocessor=preprocessor,
            preprocessor_identity=identity,
            anchor_vectors=anchor_vectors,
            anchor_identity=anchor_identity,
            authentication=authentication,
        )

    def load_frozen_model(self, source: AuthenticatedSource, seed: int) -> SemanticStatisticalContrastiveEncoder:
        if seed not in REQUIRED_SEEDS:
            raise ExperimentContractError(f"unapproved source seed: {seed}")
        model = SemanticStatisticalContrastiveEncoder(self._model_config(self.contract))
        payload = torch.load(source.checkpoint_paths[seed], map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


def _descriptor_frame(values: Any) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        missing = set(DESCRIPTOR_FIELDS) - set(values.columns)
        if missing:
            raise ExperimentContractError(f"raw descriptor fields missing: {sorted(missing)}")
        return values.loc[:, DESCRIPTOR_FIELDS]
    return pd.DataFrame(_descriptor_array(values), columns=DESCRIPTOR_FIELDS)


def _descriptor_array(values: Any) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        output = values.loc[:, DESCRIPTOR_FIELDS].to_numpy(dtype=np.float64)
    else:
        output = np.asarray(values, dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != len(DESCRIPTOR_FIELDS):
        raise ExperimentContractError("raw descriptors must have shape (N, 13)")
    if not np.isfinite(output).all():
        raise ExperimentContractError("raw descriptors contain NaN/Inf")
    return output


def _row_normalize(values: np.ndarray, label: str = "representation") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ExperimentContractError(f"{label} has invalid shape or NaN/Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ExperimentContractError(f"{label} has a zero-norm row")
    return (array / norms).astype(np.float32)


def _normalized_vector(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != JOINT_DIMENSION or not np.isfinite(array).all():
        raise ExperimentContractError(f"{label} must be a finite 32-vector")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ExperimentContractError(f"{label} is zero norm")
    return (array / norm).astype(np.float32)


def checkpoint_epoch_from_validation_losses(losses: Sequence[float], minimum_improvement: float) -> int:
    """Return the 1-based minimum-validation-loss checkpoint under the frozen delta rule."""

    best = math.inf
    best_epoch = 0
    for epoch, raw in enumerate(losses, start=1):
        value = float(raw)
        if not math.isfinite(value):
            raise ExperimentContractError("validation losses must be finite")
        if value < best - float(minimum_improvement):
            best = value
            best_epoch = epoch
    if not best_epoch:
        raise ExperimentContractError("no validation loss was supplied")
    return best_epoch


def identity_exclusions(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build symmetric exact-identity exclusions; no behavioral similarity is used."""

    required = {"feature_name", "equivalence_group_id"}
    missing = required - set(pairs)
    if missing:
        raise ExperimentContractError(f"identity exclusion fields missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for _, group in pairs.groupby("equivalence_group_id", sort=True):
        names = sorted(group["feature_name"].astype(str).tolist())
        for left in names:
            for right in names:
                if left != right:
                    rows.append(
                        {
                            "anchor_feature_name": left,
                            "excluded_feature_name": right,
                            "exclusion_reason": "exact_dev_duplicate",
                            "policy_version": "identity_equivalence_v2",
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "anchor_feature_name",
            "excluded_feature_name",
            "exclusion_reason",
            "policy_version",
        ],
    )


def _training_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("feature_name", kind="mergesort").reset_index(drop=True).copy()
    names = ordered["feature_name"].astype(str).tolist()
    order_hash = feature_order_hash(names)
    ordered["source_manifest_hash"] = ordered["source_manifest_hash"].astype(str)
    ordered["text_embedding_row_id"] = ordered["embedding_cache_key"].astype(str)
    ordered["statistical_vector_row_id"] = ordered["stable_row_id"].astype(str)
    ordered["positive_pair_index"] = range(len(ordered))
    ordered["feature_order_hash"] = order_hash
    ordered["split"] = ordered["representation_split"].astype(str)
    ordered["allowed_for_training"] = ordered["split"].eq("train")
    ordered["allowed_for_validation"] = ordered["split"].eq("validation")
    return ordered


def stability_training_bundle(package: Prompt1Package) -> TrainingDataBundle:
    pair_frame = package.pair_frame.copy()
    pair_frame["source_manifest_hash"] = package.manifest_sha256
    train = _training_pair_frame(pair_frame.loc[pair_frame["representation_split"].eq("train")])
    validation = _training_pair_frame(pair_frame.loc[pair_frame["representation_split"].eq("validation")])
    all_pairs = pd.concat([train, validation], ignore_index=True)
    exclusions = identity_exclusions(pair_frame)
    text_columns = _embedding_columns(pair_frame)
    text = pair_frame[["embedding_cache_key", "feature_name", *text_columns]].copy()
    # The legacy tensor loader's frozen wire format is stat_0000..stat_0012,
    # whereas Prompt-1 deliberately publishes human-readable stat_<field>
    # columns. Adapt names only; values and frozen field order are unchanged.
    stat = pair_frame[["stable_row_id", "feature_name"]].copy()
    for index, field in enumerate(DESCRIPTOR_FIELDS):
        stat[f"stat_{index:04d}"] = pair_frame[f"stat_{field}"].to_numpy(
            dtype=np.float32, copy=False
        )
    return TrainingDataBundle(
        train_pairs=train,
        validation_pairs=validation,
        external_pairs=pd.DataFrame(),
        source_pairs=all_pairs,
        training_text=text,
        external_text=pd.DataFrame(),
        training_stat=stat,
        external_stat=pd.DataFrame(),
        training_dataset="homecredit_model_stability_2024",
        external_dataset="none",
        negative_exclusions=exclusions,
        upstream_hashes={
            "prompt1_package_manifest_sha256": package.manifest_sha256,
            "feature_universe_hash": package.feature_universe_hash,
            "text_embedding_identity": package.text_embedding_identity,
            "statistical_preprocessor_hash": package.statistical_preprocessor_identity,
        },
        text_dim=384,
        statistical_dim=13,
        statistical_fields=list(DESCRIPTOR_FIELDS),
    )


class StabilityClipTrainer:
    """Runs five independent deterministic source-representation training seeds."""

    def __init__(
        self,
        config: Mapping[str, Any],
        config_hash: str,
        artifact_root: str | Path,
        logger: ProgressLogger,
        stage_store: StageStore,
    ) -> None:
        self.config = dict(config)
        self.config_hash = config_hash
        self.output_root = Path(artifact_root) / "representation" / "stability"
        self.logger = logger
        self.stage_store = stage_store
        implementation_files = [
            Path(__file__),
            Path(__file__).with_name("trainer.py"),
            Path(__file__).with_name("model.py"),
            Path(__file__).with_name("loss.py"),
        ]
        self.implementation_hash = canonical_hash(
            {path.name: sha256_file(path) for path in implementation_files}
        )

    def _training_config(self, package: Prompt1Package) -> ClipTrainingConfig:
        training = self.config["training"]
        architecture = dict(self.config["representation_contract"]["architecture"])
        architecture.pop("expected_parameter_count")
        placeholder = package.root / "methodology_lock.json"
        return ClipTrainingConfig(
            tensor_schema_path=placeholder,
            contrastive_pair_manifest_path=package.root / "pairs/stability_source_pairs_manifest.json",
            train_pairs_path=package.root / "pairs/stability_source_pairs.parquet",
            validation_pairs_path=package.root / "pairs/stability_source_pairs.parquet",
            external_pairs_path=placeholder,
            negative_exclusion_pairs_path=package.root / "pairing/identity_equivalence.csv",
            negative_policy_manifest_path=package.root / "methodology_lock.json",
            homecredit_text_embeddings_path=package.root / "text/text_embeddings.parquet",
            lendingclub_v2_text_embeddings_path=placeholder,
            homecredit_statistical_vectors_path=package.root / "statistics/statistical_descriptors_stability_source_scaled.csv",
            lendingclub_v2_statistical_vectors_path=placeholder,
            text_embedding_manifest_path=package.root / "text/text_embedding_manifest.json",
            statistical_preprocessor_path=package.root / "statistics/stability_source_stat_preprocessor.json",
            source_manifest_path=package.root / "manifests/data_provenance.json",
            split_manifest_path=package.root / "pairing/representation_split.csv",
            output_dir=self.output_root,
            model=ClipModelConfig(**architecture),
            optimizer="AdamW",
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            minimum_improvement=float(training["minimum_improvement"]),
            gradient_clipping_enabled=True,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            seeds=tuple(int(seed) for seed in training["seeds"]),
            deterministic=bool(training["deterministic_algorithms"]),
            device_policy=str(training["device_policy"]),
            selection_metric="validation_loss",
            collapse_thresholds=dict(training["collapse_thresholds"]),
            statistical_view_scope="compact_target_free_v2_dev_representation_train_only",
            smoke_test_steps=3,
            training_dataset="homecredit_model_stability_2024",
            external_dataset="none",
            configuration_hash=self.config_hash,
            data_manifest_hash=package.manifest_sha256,
            statistical_preprocessor_hash=package.statistical_preprocessor_identity,
            source_anchor_hash=sha256_file(package.root / "anchor/stability_source_anchor_manifest.json"),
        )

    def run(self, package: Prompt1Package) -> tuple[dict[int, SeedTrainingResult], Path]:
        _require_equal(tuple(self.config["training"]["seeds"]), REQUIRED_SEEDS, "Stability seed set")
        data = stability_training_bundle(package)
        data.upstream_hashes["stability_clip_implementation_hash"] = (
            self.implementation_hash
        )
        train_text, train_stat = tensors_for_pairs(
            data.train_pairs, data.training_text, data.training_stat
        )
        validation_text, validation_stat = tensors_for_pairs(
            data.validation_pairs, data.training_text, data.training_stat
        )
        expected_shapes = {
            "train_text": (len(data.train_pairs), 384),
            "train_stat": (len(data.train_pairs), 13),
            "validation_text": (len(data.validation_pairs), 384),
            "validation_stat": (len(data.validation_pairs), 13),
        }
        observed_shapes = {
            "train_text": tuple(train_text.shape),
            "train_stat": tuple(train_stat.shape),
            "validation_text": tuple(validation_text.shape),
            "validation_stat": tuple(validation_stat.shape),
        }
        _require_equal(observed_shapes, expected_shapes, "Stability CLIP tensor adapter")
        if not all(torch.isfinite(value).all().item() for value in (
            train_text, train_stat, validation_text, validation_stat
        )):
            raise ExperimentContractError("Stability CLIP adapted tensors contain NaN/Inf")
        training_config = self._training_config(package)
        results: dict[int, SeedTrainingResult] = {}
        snapshot = json.dumps(self.config, indent=2, sort_keys=True)
        for position, seed in enumerate(REQUIRED_SEEDS, start=1):
            stage = f"stability_seed_{seed}"
            seed_dir = self.output_root / "seeds" / f"seed_{seed}"
            marker = seed_dir / "stage_complete.json"
            inputs = {
                "config_hash": self.config_hash,
                "prompt1_package_manifest_sha256": package.manifest_sha256,
                "implementation_hash": self.implementation_hash,
            }
            if marker.exists():
                complete = _read_json(marker)
                _require_equal(complete.get("status"), "COMPLETE", f"seed {seed} status")
                _require_equal(complete.get("config_hash"), self.config_hash, f"seed {seed} config")
                _require_equal(
                    complete.get("input_hashes"), inputs, f"seed {seed} input hashes"
                )
                checkpoint = seed_dir / "best_checkpoint.pt"
                checkpoint_hash = _require_hash(checkpoint, complete["checkpoint_sha256"], f"seed {seed} checkpoint")
                log = _read_json(seed_dir / "training_log.json")
                manifest = _read_json(seed_dir / "checkpoint_manifest.json")
                results[seed] = SeedTrainingResult(
                    seed=seed,
                    best_epoch=int(log["best_epoch"]),
                    final_epoch=int(log["final_epoch"]),
                    early_stopping_epoch=int(log["early_stopping_epoch"]),
                    best_validation_loss=float(manifest["validation_value"]),
                    best_validation_mrr=float(_read_json(seed_dir / "representation_metrics.json")["validation_retrieval"]["mean_reciprocal_rank"]),
                    checkpoint_path=checkpoint,
                    checkpoint_manifest_path=seed_dir / "checkpoint_manifest.json",
                    checkpoint_hash=checkpoint_hash,
                    parameter_count=int(manifest["parameter_count"]),
                    epoch_metrics_path=seed_dir / "epoch_metrics.csv",
                    representation_metrics_path=seed_dir / "representation_metrics.json",
                    training_log_path=seed_dir / "training_log.json",
                )
                self.logger.emit(f"[CLIP][Stability][seed {seed}][{position}/5] reusing authenticated COMPLETE seed", stage="clip_training", event="seed_reuse", seed=seed)
                continue
            archived = self.stage_store.archive_incomplete_seed(seed_dir)
            if archived is not None:
                self.logger.emit(f"[CLIP][Stability][seed {seed}] archived interrupted directory {archived.name}", stage="clip_training", event="seed_archive", seed=seed)
            self.logger.emit(f"[CLIP][Stability][seed {seed}][{position}/5] starting", stage="clip_training", event="seed_start", seed=seed)
            def seed_progress(payload: Mapping[str, Any]) -> None:
                if payload.get("event") != "seed_start":
                    self.logger.callback(payload)

            result = train_seed(
                config=training_config,
                data=data,
                seed=seed,
                output_dir=self.output_root,
                config_snapshot_text=snapshot,
                progress_callback=seed_progress,
                direction="stability_to_stability",
                batch_log_interval=int(self.config["training"]["batch_log_interval"]),
            )
            metrics = _read_json(result.representation_metrics_path)["validation_retrieval"]
            atomic_json(
                marker,
                {
                    "schema_version": "stability_clip_seed_complete_v1",
                    "status": "COMPLETE",
                    "config_hash": self.config_hash,
                    "input_hashes": inputs,
                    "seed": seed,
                    "best_epoch": result.best_epoch,
                    "stop_epoch": result.final_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "mrr": metrics["mean_reciprocal_rank"],
                    "recall_at_1": (metrics["text_to_statistical_recall_at_1"] + metrics["statistical_to_text_recall_at_1"]) / 2,
                    "recall_at_5": (metrics["text_to_statistical_recall_at_5"] + metrics["statistical_to_text_recall_at_5"]) / 2,
                    "recall_at_10": (metrics["text_to_statistical_recall_at_10"] + metrics["statistical_to_text_recall_at_10"]) / 2,
                    "checkpoint_sha256": result.checkpoint_hash,
                },
            )
            results[seed] = result
        summary_rows = []
        for seed, result in results.items():
            validation = _read_json(result.representation_metrics_path)["validation_retrieval"]
            summary_rows.append({
                "seed": seed,
                "best_epoch": result.best_epoch,
                "stop_epoch": result.final_epoch,
                "best_validation_loss": result.best_validation_loss,
                "mrr": validation["mean_reciprocal_rank"],
                "recall_at_1": (validation["text_to_statistical_recall_at_1"] + validation["statistical_to_text_recall_at_1"]) / 2,
                "recall_at_5": (validation["text_to_statistical_recall_at_5"] + validation["statistical_to_text_recall_at_5"]) / 2,
                "recall_at_10": (validation["text_to_statistical_recall_at_10"] + validation["statistical_to_text_recall_at_10"]) / 2,
                "checkpoint_sha256": result.checkpoint_hash,
            })
        summary_path = atomic_csv(self.output_root / "training_summary.csv", pd.DataFrame(summary_rows))
        return results, summary_path


class FiveSeedConsensusBuilder:
    """Deterministic seed-11-reference orthogonal Procrustes consensus."""

    def __init__(self, required_seeds: Sequence[int] = REQUIRED_SEEDS, reference_seed: int = REFERENCE_SEED) -> None:
        _require_equal(tuple(required_seeds), REQUIRED_SEEDS, "consensus seed order")
        _require_equal(int(reference_seed), REFERENCE_SEED, "consensus reference seed")
        self.seeds = tuple(required_seeds)
        self.reference_seed = reference_seed

    def build(
        self,
        feature_ids: Sequence[str],
        seed_matrices: Mapping[int, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        identities = [str(value) for value in feature_ids]
        if len(identities) != len(set(identities)):
            raise ExperimentContractError("consensus feature identities are not unique")
        _require_equal(set(seed_matrices), set(self.seeds), "consensus seed presence")
        reference = _row_normalize(seed_matrices[self.reference_seed], "seed 11 representation")
        if reference.shape != (len(identities), JOINT_DIMENSION):
            raise ExperimentContractError("seed 11 representation shape mismatch")
        aligned: list[np.ndarray] = []
        rotations: dict[str, str] = {}
        for seed in self.seeds:
            current = _row_normalize(seed_matrices[seed], f"seed {seed} representation")
            if current.shape != reference.shape:
                raise ExperimentContractError(f"seed {seed} feature order/dimension mismatch")
            if seed != self.reference_seed:
                left, _, right = np.linalg.svd(current.T @ reference, full_matrices=False)
                rotation = left @ right
                current = _row_normalize(current @ rotation, f"seed {seed} aligned representation")
                rotations[str(seed)] = hashlib.sha256(np.ascontiguousarray(rotation).tobytes()).hexdigest()
            else:
                rotations[str(seed)] = "identity_reference"
            aligned.append(current)
        consensus = _row_normalize(np.mean(np.stack(aligned, axis=0), axis=0), "five-seed consensus")
        manifest = {
            "schema_version": "five_seed_procrustes_consensus_v1",
            "required_seeds": list(self.seeds),
            "reference_seed": self.reference_seed,
            "alignment": "orthogonal_procrustes_svd",
            "aggregation": "l2_normalize_each_align_mean_l2_normalize",
            "feature_count": len(identities),
            "dimension": JOINT_DIMENSION,
            "feature_order_sha256": canonical_hash(identities),
            "rotation_hashes": rotations,
            "consensus_sha256": hashlib.sha256(np.ascontiguousarray(consensus).tobytes()).hexdigest(),
        }
        return consensus, manifest


def project_joint(
    model: SemanticStatisticalContrastiveEncoder,
    text_values: np.ndarray,
    statistical_values: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ExperimentContractError("projection model must be frozen")
    # Parquet-backed pandas blocks may expose read-only NumPy views. PyTorch
    # rejects those as unsafe even for inference, so make bounded writable
    # C-contiguous projection buffers (1,959 x 384 and 1,959 x 13 at most).
    text = np.array(text_values, dtype=np.float32, order="C", copy=True)
    stats = np.array(statistical_values, dtype=np.float32, order="C", copy=True)
    if text.shape[0] != stats.shape[0] or text.shape[1] != 384 or stats.shape[1] != 13:
        raise ExperimentContractError("projection view dimensions are invalid")
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(text), batch_size):
            semantic, statistical = model(
                torch.as_tensor(text[start : start + batch_size]),
                torch.as_tensor(stats[start : start + batch_size]),
            )
            joint = torch.nn.functional.normalize((semantic + statistical) / 2.0, p=2, dim=-1)
            rows.append(joint.cpu().numpy())
    return _row_normalize(np.vstack(rows), "projected joint representation")


class TransferredRankingBuilder:
    """Builds the three frozen target-free Stability feature rankings."""

    def __init__(self, artifact_root: str | Path, feature_universe_hash: str) -> None:
        self.root = Path(artifact_root)
        self.feature_universe_hash = feature_universe_hash
        self.consensus_builder = FiveSeedConsensusBuilder()

    @staticmethod
    def _views(package: Prompt1Package) -> tuple[list[str], list[str], np.ndarray, pd.DataFrame]:
        ordered = package.pair_frame.sort_values("feature_id", kind="mergesort").reset_index(drop=True)
        text_columns = _embedding_columns(ordered)
        return (
            ordered["feature_id"].astype(str).tolist(),
            ordered["feature_name"].astype(str).tolist(),
            ordered[text_columns].to_numpy(dtype=np.float32),
            ordered,
        )

    def build_native(
        self,
        package: Prompt1Package,
        checkpoint_results: Mapping[int, SeedTrainingResult],
        model_config: ClipModelConfig,
    ) -> tuple[Path, Path]:
        feature_ids, names, text, ordered = self._views(package)
        stats = ordered[[f"stat_{field}" for field in DESCRIPTOR_FIELDS]].to_numpy(dtype=np.float32)
        seed_matrices: dict[int, np.ndarray] = {}
        checkpoint_hashes = {}
        for seed in REQUIRED_SEEDS:
            result = checkpoint_results[seed]
            _require_hash(result.checkpoint_path, result.checkpoint_hash, f"Stability seed {seed}")
            payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
            model = SemanticStatisticalContrastiveEncoder(model_config)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            seed_matrices[seed] = project_joint(model, text, stats)
            checkpoint_hashes[str(seed)] = result.checkpoint_hash
        consensus, manifest = self.consensus_builder.build(feature_ids, seed_matrices)
        by_id = {feature_id: index for index, feature_id in enumerate(feature_ids)}
        anchor_indexes = [by_id[value] for value in package.anchor_feature_ids]
        anchor = _normalized_vector(consensus[anchor_indexes].mean(axis=0), "Stability consensus anchor")
        scores = consensus @ anchor
        anchor_identity = hashlib.sha256(np.ascontiguousarray(anchor).tobytes()).hexdigest()
        ranking = self._ranking(
            feature_ids,
            names,
            scores,
            direction="stability_to_stability",
            source_dataset="homecredit_model_stability_2024",
            source_anchor_identity=anchor_identity,
            source_preprocessor_identity=package.statistical_preprocessor_identity,
        )
        path = atomic_csv(self.root / "rankings/stability_to_stability.csv", ranking)
        manifest.update({
            "checkpoint_hashes": checkpoint_hashes,
            "source_anchor_identity": anchor_identity,
            "source_anchor_member_ids": list(package.anchor_feature_ids),
            "target_used": False,
            "oot_used": False,
            "ranking_sha256": sha256_file(path),
        })
        manifest_path = atomic_json(self.root / "representation/stability/consensus_manifest.json", manifest)
        return path, manifest_path

    def build_transfer(
        self,
        package: Prompt1Package,
        source: AuthenticatedSource,
        resolver: HistoricalSourceArtifactResolver,
    ) -> tuple[Path, Path]:
        feature_ids, names, text, ordered = self._views(package)
        raw_by_id = package.raw_descriptors.assign(
            feature_id=package.raw_descriptors["feature_id"].astype(str)
        ).set_index("feature_id")
        raw = raw_by_id.loc[feature_ids, list(DESCRIPTOR_FIELDS)].reset_index(drop=True)
        transformed = source.preprocessor.transform(raw)
        seed_matrices: dict[int, np.ndarray] = {}
        seed_scores: dict[int, np.ndarray] = {}
        for seed in REQUIRED_SEEDS:
            model = resolver.load_frozen_model(source, seed)
            seed_matrices[seed] = project_joint(model, text, transformed)
            seed_scores[seed] = seed_matrices[seed] @ source.anchor_vectors[seed]
        consensus, consensus_manifest = self.consensus_builder.build(feature_ids, seed_matrices)
        if source.name == "homecredit":
            # Corrected HC uses the authenticated consensus-space source anchor.
            scores = consensus @ source.anchor_vectors[REFERENCE_SEED]
            score_rule = "procrustes_consensus_then_frozen_homecredit_consensus_anchor_similarity"
        else:
            # Corrected LC reverse transfer ranks by arithmetic mean of per-seed
            # similarity to the corresponding frozen source anchor.
            scores = np.mean(np.stack([seed_scores[seed] for seed in REQUIRED_SEEDS]), axis=0)
            score_rule = "arithmetic_mean_of_five_frozen_seed_anchor_similarities"
        direction = f"{source.name}_to_stability"
        ranking = self._ranking(
            feature_ids,
            names,
            scores,
            direction=direction,
            source_dataset=source.dataset,
            source_anchor_identity=source.anchor_identity,
            source_preprocessor_identity=source.preprocessor_identity,
        )
        path = atomic_csv(self.root / f"rankings/{direction}.csv", ranking)
        manifest = {
            **consensus_manifest,
            "schema_version": "transferred_ranking_manifest_v1",
            "direction": direction,
            "score_rule": score_rule,
            "checkpoint_hashes": {str(key): value for key, value in source.checkpoint_hashes.items()},
            "source_anchor_identity": source.anchor_identity,
            "source_preprocessor_identity": source.preprocessor_identity,
            "target_used": False,
            "oot_used": False,
            "ranking_sha256": sha256_file(path),
        }
        manifest_path = atomic_json(self.root / f"manifests/{direction}_projection_consensus.json", manifest)
        return path, manifest_path

    def _ranking(
        self,
        feature_ids: Sequence[str],
        names: Sequence[str],
        scores: np.ndarray,
        *,
        direction: str,
        source_dataset: str,
        source_anchor_identity: str,
        source_preprocessor_identity: str,
    ) -> pd.DataFrame:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(values) != FEATURE_COUNT or not np.isfinite(values).all():
            raise ExperimentContractError(f"{direction} did not score exactly 1,959 finite identities")
        frame = pd.DataFrame({"feature_id": feature_ids, "feature_name": names, "clip_score": values})
        if not frame["feature_id"].is_unique or not frame["feature_name"].is_unique:
            raise ExperimentContractError(f"{direction} ranking identities are not one-to-one")
        frame = frame.sort_values(["clip_score", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))
        frame["direction"] = direction
        frame["source_dataset"] = source_dataset
        frame["target_dataset"] = "homecredit_model_stability_2024"
        frame["feature_universe_hash"] = self.feature_universe_hash
        frame["consensus_method"] = "five_seed_seed11_reference_orthogonal_procrustes"
        frame["source_anchor_identity"] = source_anchor_identity
        frame["source_preprocessor_identity"] = source_preprocessor_identity
        return frame


class _ProgressQueue:
    def __init__(self, logger: ProgressLogger, event: str) -> None:
        self.logger = logger
        self.event = event

    def put(self, payload: Mapping[str, Any]) -> None:
        metrics = {key: value for key, value in payload.items() if key not in {"stage", "fold_id"}}
        self.logger.emit(
            f"[MATRIX] {payload.get('stage')} {payload.get('fold_id')} {metrics}",
            stage=str(payload.get("stage", "matrix_read")),
            event=self.event,
            metrics=metrics,
        )


@dataclass(frozen=True)
class OOTAccessToken:
    manifest_path: Path
    manifest_sha256: str
    config_hash: str


class StabilityMatrixAccess:
    """Candidate-restricted authenticated matrix reader with a hard OOT gate."""

    def __init__(self, repository_root: str | Path, config: Mapping[str, Any], logger: ProgressLogger) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.config = dict(config)
        self.logger = logger
        self.matrix_root = _resolve(self.repository_root, self.config["matrix_root"])
        self.protocol_path = _resolve(self.repository_root, self.config["protocol_lock"])
        _require_hash(self.matrix_root / "manifest.json", self.config["matrix_manifest_sha256"], "Stability matrix manifest")
        _require_hash(self.matrix_root / "metadata.json", self.config["matrix_metadata_sha256"], "Stability matrix metadata")
        _require_hash(self.protocol_path, self.config["protocol_lock_sha256"], "Stability temporal protocol")
        self.manifest, self.metadata = _matrix_identity(self.matrix_root)
        self.protocol = _read_json(self.protocol_path)
        self.split = self.protocol["approved_protocol"]["split_and_fold_boundaries"]
        _require_equal(int(self.split["dev"]["rows"]), int(self.config["dev_rows"]), "DEV row count lock")
        _require_equal(int(self.split["oot"]["rows"]), int(self.config["oot_rows"]), "OOT row count lock")
        _require_equal(self.split["oot_start_inclusive"], self.config["oot_start_inclusive"], "OOT boundary")
        predictors = list(self.metadata["predictor_columns"])
        if len(predictors) != FEATURE_COUNT or len(set(predictors)) != FEATURE_COUNT:
            raise ExperimentContractError("matrix predictor identity is not exactly 1,959 unique columns")
        self.predictors = set(predictors)

    def _validate_predictors(self, predictors: Sequence[str], expected_count: int | None = None) -> list[str]:
        columns = [str(value) for value in predictors]
        if len(columns) != len(set(columns)) or not set(columns).issubset(self.predictors):
            raise ExperimentContractError("candidate-restricted matrix columns are invalid")
        if set(columns) & set(NON_PREDICTORS):
            raise ExperimentContractError("non-predictor requested as a modeling feature")
        if expected_count is not None and len(columns) != expected_count:
            raise ExperimentContractError(f"candidate restriction must be exactly {expected_count}, found {len(columns)}")
        return columns

    def _read(self, expected: Mapping[str, Any], predictors: Sequence[str], label: str) -> pd.DataFrame:
        frame = _read_date_slice(
            self.matrix_root,
            self.manifest,
            date_min=str(expected["date_min"]),
            date_max=str(expected["date_max"]),
            predictors=predictors,
            stop_event=None,
            stage_queue=_ProgressQueue(self.logger, "matrix_part"),
            stage="clip_downstream_matrix_read",
            fold_label=label,
        )
        _validate_scope_frame(frame, expected, label)
        return frame

    def load_fold(self, fold_id: int, predictors: Sequence[str], expected_pool_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = self._validate_predictors(predictors, expected_pool_size)
        fold = next((value for value in self.split["folds"] if int(value["fold_id"]) == int(fold_id)), None)
        if fold is None:
            raise ExperimentContractError(f"temporal fold is not locked: {fold_id}")
        train = self._read(fold["train"], columns, f"fold_{fold_id}:train")
        validation = self._read(fold["validation"], columns, f"fold_{fold_id}:validation")
        if set(pd.to_datetime(train["date_decision"]).dt.normalize()) & set(pd.to_datetime(validation["date_decision"]).dt.normalize()):
            raise ExperimentContractError(f"fold {fold_id} train/validation dates overlap")
        if set(train["case_id"]) & set(validation["case_id"]):
            raise ExperimentContractError(f"fold {fold_id} train/validation case IDs overlap")
        return train, validation

    def load_full_dev(self, predictors: Sequence[str], expected_pool_size: int) -> pd.DataFrame:
        columns = self._validate_predictors(predictors, expected_pool_size)
        return self._read(self.split["dev"], columns, "full_dev")

    def load_oot(
        self,
        predictors: Sequence[str],
        token: OOTAccessToken | None,
        gate: "PreOOTFreezeGate",
    ) -> pd.DataFrame:
        if token is None:
            raise ExperimentContractError("OOT access rejected: pre-OOT freeze token is absent")
        gate.validate(token)
        columns = self._validate_predictors(predictors)
        frame = self._read(self.split["oot"], columns, "oot")
        if len(frame) != int(self.config["oot_rows"]):
            raise ExperimentContractError("OOT row count mismatch; no row dropping is permitted")
        return frame


@dataclass(frozen=True)
class DownstreamCell:
    direction: str
    classifier: str
    experiment_id: str
    pool_size: int
    final_k: int


def frozen_downstream_cells(config: Mapping[str, Any]) -> list[DownstreamCell]:
    expected_directions = ("stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability")
    observed = tuple(item["id"] for item in config["directions"])
    _require_equal(observed, expected_directions, "downstream direction set")
    cells = []
    for direction in expected_directions:
        for classifier in ("lr", "catboost"):
            contract = config["downstream"]["cells"][classifier]
            cells.append(
                DownstreamCell(
                    direction=direction,
                    classifier=classifier,
                    experiment_id=f"{config['experiment_id']}__{direction}__{classifier}",
                    pool_size=int(contract["candidate_pool_size"]),
                    final_k=int(contract["final_k"]),
                )
            )
    _require_equal([(c.classifier, c.pool_size, c.final_k) for c in cells[0:2]], [("lr", 60, 20), ("catboost", 100, 40)], "downstream pool/budget lock")
    return cells


def _predict_batches(model: Any, matrix: Any, batch_size: int, callback: Callable[[int, int], None] | None = None) -> np.ndarray:
    values: list[np.ndarray] = []
    total = math.ceil(matrix.shape[0] / batch_size)
    for index, start in enumerate(range(0, matrix.shape[0], batch_size), start=1):
        stop = min(start + batch_size, matrix.shape[0])
        prediction = np.asarray(model.predict_proba(matrix[start:stop])[:, 1], dtype=np.float64)
        values.append(prediction)
        if callback is not None:
            callback(index, total)
    return np.concatenate(values) if values else np.array([], dtype=np.float64)


def _new_classifier(name: str, config: Mapping[str, Any]) -> Any:
    if name == "lr":
        return LogisticRegression(**dict(config["models"]["lr"]))
    if name != "catboost":
        raise ExperimentContractError(f"unsupported classifier: {name}")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ExperimentContractError("CatBoost is required for the frozen CatBoost cells") from exc
    options = dict(config["models"]["catboost"])
    # The config's verbose cadence is observational. No eval_set or early stop is used.
    return CatBoostClassifier(**options)


def _atomic_joblib(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _selection_evidence(selector: RandomForestRelevanceMRMRSelector, ranking: pd.DataFrame, fold_id: int | str) -> pd.DataFrame:
    selected = list(selector.selected_features_ or [])
    by_name = ranking.set_index("feature_name")
    trace = getattr(selector, "selection_trace_", pd.DataFrame()).set_index("feature") if hasattr(selector, "selection_trace_") else pd.DataFrame()
    rows = []
    for order, name in enumerate(selected, start=1):
        item = {
            "fold_id": fold_id,
            "feature_name": name,
            "clip_rank": int(by_name.loc[name, "rank"]),
            "clip_score": float(by_name.loc[name, "clip_score"]),
            "legacy_mrmr_selection_order": order,
            "rf_relevance": float(selector.rf_importances_.loc[name]),
            "mrmr_method": selector.algorithm_name,
        }
        if len(trace) and name in trace.index:
            item["mean_absolute_correlation"] = float(trace.loc[name, "mean_absolute_correlation"])
            item["selection_score"] = float(trace.loc[name, "selection_score"])
        rows.append(item)
    return pd.DataFrame(rows)


class LegacyMRMRDownstreamRunner:
    """Frozen candidate pool -> fold/full-DEV mRMR -> Stability classifier."""

    def __init__(
        self,
        config: Mapping[str, Any],
        artifact_root: str | Path,
        result_root: str | Path,
        matrix: StabilityMatrixAccess,
        logger: ProgressLogger,
    ) -> None:
        self.config = dict(config)
        self.downstream = self.config["downstream"]
        self.artifact_root = Path(artifact_root)
        self.result_root = Path(result_root)
        self.matrix = matrix
        self.logger = logger
        self.cells = frozen_downstream_cells(self.config)
        _require_equal(self.downstream["mrmr"]["implementation"], "credit_risk_fs.selectors.mrmr.RandomForestRelevanceMRMRSelector", "legacy mRMR implementation")
        _require_equal(RandomForestRelevanceMRMRSelector.algorithm_name, "rf_relevance_correlation_redundancy", "legacy mRMR identity")

    def _ranking(self, direction: str) -> pd.DataFrame:
        path = self.artifact_root / "rankings" / f"{direction}.csv"
        ranking = pd.read_csv(path)
        if len(ranking) != FEATURE_COUNT or ranking["rank"].astype(int).tolist() != list(range(1, FEATURE_COUNT + 1)):
            raise ExperimentContractError(f"invalid frozen ranking for {direction}")
        return ranking

    def _candidate_pool(self, cell: DownstreamCell, ranking: pd.DataFrame) -> tuple[pd.DataFrame, list[str], Path]:
        pool = ranking.head(cell.pool_size).copy()
        if len(pool) != cell.pool_size:
            raise ExperimentContractError(f"{cell.experiment_id} candidate pool size changed")
        pool["candidate_pool_size"] = cell.pool_size
        pool["final_k"] = cell.final_k
        path = atomic_csv(self._cell_root(cell) / "candidate_pool.csv", pool)
        return pool, pool["feature_name"].astype(str).tolist(), path

    def _cell_root(self, cell: DownstreamCell) -> Path:
        return self.result_root / "downstream" / cell.direction / cell.classifier

    def _fit_selector(self, raw: pd.DataFrame, target: pd.Series, cell: DownstreamCell, fold: int | str) -> tuple[RandomForestRelevanceMRMRSelector, OriginalFeatureNumericEncoder]:
        if raw.shape[1] != cell.pool_size:
            raise ExperimentContractError(f"{cell.experiment_id} mRMR saw {raw.shape[1]} instead of {cell.pool_size} candidates")
        encoder = OriginalFeatureNumericEncoder()
        with self.logger.heartbeat(
            f"[DOWNSTREAM][{cell.direction}][{cell.classifier}][fold {fold}] selection encoding/RF relevance/correlation redundancy running",
            stage="legacy_mrmr",
            direction=cell.direction,
            fold=int(fold) if isinstance(fold, int) else None,
            interval=float(self.downstream["long_operation_heartbeat_seconds"]),
        ):
            encoded = encoder.fit_transform(raw)
            selector = RandomForestRelevanceMRMRSelector(
                k=cell.final_k,
                method="mrmr",
                correlation="pearson",
                random_state=int(self.downstream["mrmr"]["random_state"]),
                n_jobs=int(self.downstream["mrmr"]["n_jobs"]),
            )
            selector.fit(encoded, target.reset_index(drop=True))
        if len(selector.selected_features_ or []) != cell.final_k:
            raise ExperimentContractError(f"{cell.experiment_id} mRMR did not select K={cell.final_k}")
        return selector, encoder

    def run_dev(self) -> tuple[pd.DataFrame, list[Path]]:
        summaries = []
        outputs: list[Path] = []
        for cell in self.cells:
            ranking = self._ranking(cell.direction)
            _, predictors, candidate_path = self._candidate_pool(cell, ranking)
            outputs.append(candidate_path)
            fold_metrics = []
            fold_evidence = []
            oof_rows = []
            previous_selected: set[str] | None = None
            for fold_id in range(1, 6):
                fold_started = time.perf_counter()
                self.logger.emit(
                    f"[DOWNSTREAM][{cell.direction}][{cell.classifier}][fold {fold_id}/5] candidate_pool={cell.pool_size} mrmr_target_k={cell.final_k}",
                    stage="dev_downstream",
                    event="fold_start",
                    direction=cell.direction,
                    fold=fold_id,
                )
                train, validation = self.matrix.load_fold(fold_id, predictors, cell.pool_size)
                target_train = train["target"].astype(int).copy()
                target_validation = validation["target"].astype(int).copy()
                selector, _ = self._fit_selector(train[predictors], target_train, cell, fold_id)
                selected = list(selector.selected_features_)
                fold_evidence.append(_selection_evidence(selector, ranking, fold_id))
                preprocessor = SparsePreprocessor()
                with self.logger.heartbeat(
                    f"[DOWNSTREAM][{cell.direction}][{cell.classifier}][fold {fold_id}/5] preprocessor/classifier fit running",
                    stage="classifier_fit",
                    direction=cell.direction,
                    fold=fold_id,
                    interval=float(self.downstream["long_operation_heartbeat_seconds"]),
                ):
                    train_matrix = preprocessor.fit_transform(train[selected])
                    validation_matrix = preprocessor.transform(validation[selected])
                    classifier = _new_classifier(cell.classifier, self.downstream)
                    # No validation values or labels enter fit; CatBoost gets no eval_set.
                    if cell.classifier == "catboost":
                        classifier.fit(train_matrix, target_train, eval_set=None)
                    else:
                        classifier.fit(train_matrix, target_train)
                score_batch_size = int(self.downstream["score_batch_size"])
                train_prediction = _predict_batches(classifier, train_matrix, score_batch_size)
                threshold = determine_threshold(target_train, train_prediction)
                validation_prediction = _predict_batches(classifier, validation_matrix, score_batch_size)
                metrics = evaluate_model(target_validation, validation_prediction, threshold=threshold)
                current_selected = set(selected)
                metrics.update({
                    "fold_id": fold_id,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "threshold_fit_scope": "fold_train_only",
                    "selected_feature_jaccard_vs_previous_fold": (
                        None if previous_selected is None else jaccard_similarity(previous_selected, current_selected)
                    ),
                    "elapsed_seconds": time.perf_counter() - fold_started,
                    **_resource_snapshot(),
                })
                previous_selected = current_selected
                fold_metrics.append(metrics)
                oof_rows.append(pd.DataFrame({
                    "case_id": validation["case_id"].to_numpy(),
                    "date_decision": validation["date_decision"].to_numpy(),
                    "fold_id": fold_id,
                    "target": target_validation.to_numpy(),
                    "prediction": validation_prediction,
                    "direction": cell.direction,
                    "classifier": cell.classifier,
                    "experiment_id": cell.experiment_id,
                }))
                self.logger.emit(
                    f"[DOWNSTREAM][{cell.direction}][{cell.classifier}][fold {fold_id}/5] fold_auc={metrics['auc']:.8f}",
                    stage="dev_downstream",
                    event="fold_end",
                    direction=cell.direction,
                    fold=fold_id,
                    metrics={"fold_auc": metrics["auc"], "selected": len(selected)},
                )
                del train, validation, train_matrix, validation_matrix, classifier, preprocessor
                gc.collect()
            oof = pd.concat(oof_rows, ignore_index=True)
            expected_validation_rows = sum(int(item["validation"]["rows"]) for item in self.matrix.split["folds"])
            if len(oof) != expected_validation_rows or oof["case_id"].duplicated().any():
                raise ExperimentContractError(f"{cell.experiment_id} OOF rows are not exactly once per locked validation identity")
            if (pd.to_datetime(oof["date_decision"]) >= pd.Timestamp(self.downstream["oot_start_inclusive"])).any():
                raise ExperimentContractError(f"{cell.experiment_id} OOF contains OOT rows")
            fold_auc = np.array([row["auc"] for row in fold_metrics], dtype=float)
            pooled_auc = float(roc_auc_score(oof["target"], oof["prediction"]))
            cell_root = self._cell_root(cell)
            oof_path = atomic_parquet(cell_root / "dev_oof_predictions.parquet", oof)
            fold_metrics_path = atomic_csv(cell_root / "fold_metrics.csv", pd.DataFrame(fold_metrics))
            selection_path = atomic_csv(cell_root / "fold_selected_features.csv", pd.concat(fold_evidence, ignore_index=True))
            outputs.extend([oof_path, fold_metrics_path, selection_path])
            summaries.append({
                "direction": cell.direction,
                "classifier": cell.classifier,
                "experiment_id": cell.experiment_id,
                "candidate_pool_size": cell.pool_size,
                "final_k": cell.final_k,
                "dev_fold_auc_mean": float(fold_auc.mean()),
                "dev_fold_auc_sd": float(fold_auc.std(ddof=1)),
                "dev_pooled_oof_auc": pooled_auc,
                "oof_rows": len(oof),
            })
        summary_path = atomic_csv(self.result_root / "analysis/dev_clip_results.csv", pd.DataFrame(summaries))
        outputs.append(summary_path)
        return pd.DataFrame(summaries), outputs

    def fit_full_dev(self) -> list[Path]:
        outputs: list[Path] = []
        for cell in self.cells:
            ranking = self._ranking(cell.direction)
            _, predictors, candidate_path = self._candidate_pool(cell, ranking)
            outputs.append(candidate_path)
            self.logger.emit(
                f"[FREEZE][{cell.direction}][{cell.classifier}] full-DEV mRMR/preprocessor/model fit starting",
                stage="full_dev_fit",
                event="cell_start",
                direction=cell.direction,
            )
            dev = self.matrix.load_full_dev(predictors, cell.pool_size)
            target = dev["target"].astype(int).copy()
            selector, selection_encoder = self._fit_selector(dev[predictors], target, cell, "full_dev")
            selected = list(selector.selected_features_)
            selection = _selection_evidence(selector, ranking, "full_dev")
            cell_root = self._cell_root(cell)
            selection_path = atomic_csv(cell_root / "full_dev_selected_features.csv", selection)
            preprocessor = SparsePreprocessor()
            with self.logger.heartbeat(
                f"[FREEZE][{cell.direction}][{cell.classifier}] full-DEV classifier running",
                stage="full_dev_fit",
                direction=cell.direction,
                interval=float(self.downstream["long_operation_heartbeat_seconds"]),
            ):
                dev_matrix = preprocessor.fit_transform(dev[selected])
                classifier = _new_classifier(cell.classifier, self.downstream)
                if cell.classifier == "catboost":
                    classifier.fit(dev_matrix, target, eval_set=None)
                else:
                    classifier.fit(dev_matrix, target)
            training_prediction = _predict_batches(classifier, dev_matrix, int(self.downstream["score_batch_size"]))
            threshold = determine_threshold(target, training_prediction)
            score_reference_path = atomic_parquet(
                cell_root / "full_dev_score_reference.parquet",
                pd.DataFrame({"case_id": dev["case_id"].to_numpy(), "prediction": training_prediction}),
            )
            preprocessor_path = _atomic_joblib(cell_root / "frozen_full_dev_preprocessor.joblib", preprocessor)
            classifier_path = _atomic_joblib(cell_root / "frozen_full_dev_classifier.joblib", classifier)
            freeze_path = atomic_json(cell_root / "full_dev_freeze_manifest.json", {
                "schema_version": "clip_full_dev_cell_freeze_v1",
                "status": "COMPLETE",
                "direction": cell.direction,
                "classifier": cell.classifier,
                "experiment_id": cell.experiment_id,
                "fit_scope": "full_DEV_only_date_decision_before_2020-02-26",
                "rows": len(dev),
                "candidate_pool_size": cell.pool_size,
                "final_k": cell.final_k,
                "selected_features": selected,
                "decision_threshold": threshold,
                "threshold_fit_scope": "full_DEV_training_predictions_only",
                "selection_encoder": "OriginalFeatureNumericEncoder_one_to_one_original_variables",
                "mrmr_method": RandomForestRelevanceMRMRSelector.algorithm_name,
                "model_preprocessor": preprocessor.schema_metadata(),
                "model_config": self.downstream["models"][cell.classifier],
                "candidate_pool_sha256": sha256_file(candidate_path),
                "selected_features_sha256": sha256_file(selection_path),
                "preprocessor_sha256": sha256_file(preprocessor_path),
                "classifier_sha256": sha256_file(classifier_path),
                "full_dev_score_reference_sha256": sha256_file(score_reference_path),
                "resource_at_freeze": _resource_snapshot(),
                "oot_values_opened": False,
            })
            outputs.extend([selection_path, score_reference_path, preprocessor_path, classifier_path, freeze_path])
            self.logger.emit(
                f"[FREEZE][{cell.direction}][{cell.classifier}] full-DEV choices frozen",
                stage="full_dev_fit",
                event="cell_end",
                direction=cell.direction,
                metrics={"selected": len(selected), "threshold": threshold},
            )
            del dev, dev_matrix, classifier, preprocessor, selection_encoder
            gc.collect()
        return outputs


class PreOOTFreezeGate:
    """Hashes every predeclared scientific choice before allowing OOT reads."""

    def __init__(self, artifact_root: str | Path, result_root: str | Path, config_hash: str) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.result_root = Path(result_root).resolve()
        self.config_hash = config_hash
        self.path = self.artifact_root / "manifests/pre_oot_freeze_manifest.json"

    def create(
        self,
        *,
        prompt1_hash: str,
        feature_universe_hash: str,
        source_authentication_path: str | Path,
        stability_checkpoint_path: str | Path,
        consensus_paths: Sequence[str | Path],
        cells: Sequence[DownstreamCell],
        random_seeds: Sequence[int],
    ) -> OOTAccessToken:
        _require_equal(tuple(random_seeds), REQUIRED_SEEDS, "pre-OOT seed freeze")
        files: list[Path] = [Path(source_authentication_path), Path(stability_checkpoint_path)]
        files.extend(Path(path) for path in consensus_paths)
        for direction in ("stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability"):
            files.append(self.artifact_root / "rankings" / f"{direction}.csv")
        for cell in cells:
            root = self.result_root / "downstream" / cell.direction / cell.classifier
            files.extend([
                root / "candidate_pool.csv",
                root / "fold_selected_features.csv",
                root / "full_dev_selected_features.csv",
                root / "full_dev_freeze_manifest.json",
                root / "full_dev_score_reference.parquet",
                root / "frozen_full_dev_preprocessor.joblib",
                root / "frozen_full_dev_classifier.joblib",
            ])
        file_hashes = {}
        for path in files:
            if not path.is_file():
                raise ExperimentContractError(f"pre-OOT gate missing frozen artifact: {path}")
            file_hashes[path.resolve().as_posix()] = sha256_file(path)
        manifest = {
            "schema_version": "clip_pre_oot_freeze_manifest_v1",
            "status": "PASS",
            "config_hash": self.config_hash,
            "prompt1_package_manifest_sha256": prompt1_hash,
            "feature_universe_hash": feature_universe_hash,
            "directions": ["stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability"],
            "cells": [asdict(cell) for cell in cells],
            "stability_clip_seeds": list(random_seeds),
            "consensus_reference_seed": REFERENCE_SEED,
            "model_random_seed": 42,
            "mrmr_random_seed": 42,
            "threshold_contract": "maximize_KS_on_full_DEV_training_predictions_only",
            "all_full_dev_models_fitted": True,
            "oot_values_opened": False,
            "frozen_file_hashes": file_hashes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(self.path, manifest)
        token = OOTAccessToken(self.path, sha256_file(self.path), self.config_hash)
        self.validate(token)
        return token

    def validate(self, token: OOTAccessToken) -> None:
        if token.manifest_path.resolve() != self.path.resolve():
            raise ExperimentContractError("OOT gate token path mismatch")
        _require_equal(token.config_hash, self.config_hash, "OOT gate config token")
        _require_hash(self.path, token.manifest_sha256, "pre-OOT freeze manifest")
        manifest = _read_json(self.path)
        _require_equal(manifest.get("status"), "PASS", "pre-OOT gate status")
        _require_equal(manifest.get("config_hash"), self.config_hash, "pre-OOT gate config")
        _require_equal(manifest.get("all_full_dev_models_fitted"), True, "pre-OOT model freeze")
        _require_equal(manifest.get("oot_values_opened"), False, "pre-OOT access boundary")
        for path, expected in manifest.get("frozen_file_hashes", {}).items():
            _require_hash(Path(path), str(expected), "pre-OOT frozen artifact")


def _lift_capture(y_true: Sequence[int], prediction: Sequence[float], fraction: float = 0.10) -> dict[str, float]:
    frame = pd.DataFrame({"target": y_true, "prediction": prediction}).sort_values(
        "prediction", ascending=False, kind="mergesort"
    )
    count = max(1, int(math.ceil(len(frame) * fraction)))
    top = frame.head(count)
    base_rate = float(frame["target"].mean())
    capture = float(top["target"].sum() / frame["target"].sum()) if frame["target"].sum() else 0.0
    lift = float(top["target"].mean() / base_rate) if base_rate else 0.0
    return {"capture_at_10pct": capture, "lift_at_10pct": lift}


def _resource_snapshot() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return {"rss_mb": float(process.memory_info().rss / (1024**2)), "cpu_percent": float(process.cpu_percent())}
    except (ImportError, OSError):
        return {"rss_mb": None, "cpu_percent": None}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.10g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


class FinalOOTEvaluator:
    """Final-score-only evaluator; all fitted objects predate OOT access."""

    def __init__(
        self,
        config: Mapping[str, Any],
        result_root: str | Path,
        matrix: StabilityMatrixAccess,
        gate: PreOOTFreezeGate,
        logger: ProgressLogger,
    ) -> None:
        self.config = dict(config)
        self.downstream = self.config["downstream"]
        self.result_root = Path(result_root)
        self.matrix = matrix
        self.gate = gate
        self.logger = logger
        self.cells = frozen_downstream_cells(self.config)

    def run(self, token: OOTAccessToken) -> tuple[pd.DataFrame, list[Path]]:
        self.gate.validate(token)
        dev_summary = pd.read_csv(self.result_root / "analysis/dev_clip_results.csv")
        results = []
        outputs: list[Path] = []
        for cell in self.cells:
            root = self.result_root / "downstream" / cell.direction / cell.classifier
            freeze_path = root / "full_dev_freeze_manifest.json"
            freeze_hash_before = sha256_file(freeze_path)
            freeze = _read_json(freeze_path)
            _require_equal(freeze.get("fit_scope"), "full_DEV_only_date_decision_before_2020-02-26", "full-DEV fit scope")
            _require_equal(int(freeze["final_k"]), cell.final_k, "frozen final K")
            selected = [str(value) for value in freeze["selected_features"]]
            if len(selected) != cell.final_k:
                raise ExperimentContractError(f"{cell.experiment_id} selected-feature freeze changed")
            preprocessor_path = root / "frozen_full_dev_preprocessor.joblib"
            classifier_path = root / "frozen_full_dev_classifier.joblib"
            _require_hash(preprocessor_path, freeze["preprocessor_sha256"], "frozen full-DEV preprocessor")
            _require_hash(classifier_path, freeze["classifier_sha256"], "frozen full-DEV classifier")
            self.logger.emit(
                f"[OOT][{cell.direction}][{cell.classifier}] final scoring starting",
                stage="oot_scoring",
                event="cell_start",
                direction=cell.direction,
            )
            # This is the first allowed opening of any OOT feature values.
            oot = self.matrix.load_oot(selected, token, self.gate)
            preprocessor = joblib.load(preprocessor_path)
            classifier = joblib.load(classifier_path)
            with self.logger.heartbeat(
                f"[OOT][{cell.direction}][{cell.classifier}] transform/prediction running",
                stage="oot_scoring",
                direction=cell.direction,
                interval=float(self.downstream["long_operation_heartbeat_seconds"]),
            ):
                oot_matrix = preprocessor.transform(oot[selected])

                def batch_progress(index: int, total: int) -> None:
                    self.logger.emit(
                        f"[OOT][{cell.direction}][{cell.classifier}] prediction batch {index}/{total}",
                        stage="oot_scoring",
                        event="prediction_batch",
                        direction=cell.direction,
                        batch=index,
                        metrics={"batch_count": total},
                    )

                prediction = _predict_batches(
                    classifier,
                    oot_matrix,
                    int(self.downstream["score_batch_size"]),
                    batch_progress,
                )
            if len(prediction) != int(self.downstream["oot_rows"]):
                raise ExperimentContractError(f"{cell.experiment_id} silently dropped OOT rows")
            target = oot["target"].astype(int).to_numpy()
            threshold = float(freeze["decision_threshold"])
            metrics = evaluate_model(target, prediction, threshold=threshold)
            metrics.update(_lift_capture(target, prediction))
            score_reference_path = root / "full_dev_score_reference.parquet"
            _require_hash(
                score_reference_path,
                freeze["full_dev_score_reference_sha256"],
                "frozen full-DEV score reference",
            )
            score_reference = pd.read_parquet(score_reference_path, columns=["prediction"])
            metrics["score_psi"] = calculate_psi(score_reference["prediction"], pd.Series(prediction))
            metrics["resource_at_oot_score"] = _resource_snapshot()
            predictions = pd.DataFrame({
                "case_id": oot["case_id"].to_numpy(),
                "date_decision": oot["date_decision"].to_numpy(),
                "target": target,
                "prediction": prediction,
                "direction": cell.direction,
                "classifier": cell.classifier,
                "experiment_id": cell.experiment_id,
            })
            if len(predictions) != int(self.downstream["oot_rows"]) or predictions["case_id"].duplicated().any():
                raise ExperimentContractError(f"{cell.experiment_id} OOT stable row identity failed")
            prediction_path = atomic_parquet(
                self.result_root / "predictions" / cell.direction / cell.classifier / "oot_predictions.parquet",
                predictions,
            )
            metric_path = atomic_json(root / "oot_metrics.json", {
                "schema_version": "clip_stability_oot_metrics_v1",
                "status": "COMPLETE",
                "experiment_id": cell.experiment_id,
                "rows": len(predictions),
                "threshold_source": "frozen_full_DEV_training_predictions_only",
                "no_oot_tuning": True,
                **metrics,
                "prediction_sha256": sha256_file(prediction_path),
                "pre_oot_manifest_sha256": token.manifest_sha256,
            })
            outputs.extend([prediction_path, metric_path])
            dev = dev_summary.loc[
                dev_summary["direction"].eq(cell.direction)
                & dev_summary["classifier"].eq(cell.classifier)
            ].iloc[0]
            results.append({
                "direction": cell.direction,
                "classifier": cell.classifier,
                "clip_source": {
                    "stability_to_stability": "homecredit_model_stability_2024",
                    "homecredit_to_stability": "homecredit",
                    "lendingclub_to_stability": "lendingclub_v2",
                }[cell.direction],
                "target_dataset": "homecredit_model_stability_2024",
                "feature_universe_count": FEATURE_COUNT,
                "candidate_pool_size": cell.pool_size,
                "final_k": cell.final_k,
                "dev_fold_auc_mean": float(dev["dev_fold_auc_mean"]),
                "dev_fold_auc_sd": float(dev["dev_fold_auc_sd"]),
                "dev_pooled_oof_auc": float(dev["dev_pooled_oof_auc"]),
                "oot_auc": metrics["auc"],
                "oot_gini": metrics["gini"],
                "oot_ks": metrics["ks"],
                "oot_lift_at_10pct": metrics["lift_at_10pct"],
                "oot_capture_at_10pct": metrics["capture_at_10pct"],
                "oot_precision": metrics["precision"],
                "oot_recall": metrics["recall"],
                "oot_f1": metrics["f1"],
                "oot_accuracy": metrics["accuracy"],
                "oot_log_loss": metrics["log_loss"],
                "oot_brier": metrics["brier"],
                "oot_score_psi": metrics["score_psi"],
                "representation_checkpoint_rule": "minimum_source_validation_loss_per_seed",
                "consensus_rule": "five_seed_seed11_reference_orthogonal_procrustes",
                "mrmr_method": RandomForestRelevanceMRMRSelector.algorithm_name,
                "status": "COMPLETE",
            })
            _require_equal(sha256_file(freeze_path), freeze_hash_before, "full-DEV freeze after OOT")
            self.logger.emit(
                f"[OOT][{cell.direction}][{cell.classifier}] oot_auc={metrics['auc']:.8f} complete",
                stage="oot_scoring",
                event="cell_end",
                direction=cell.direction,
                metrics={"oot_auc": metrics["auc"], "oot_rows": len(predictions)},
            )
            del oot, oot_matrix, preprocessor, classifier, predictions
            gc.collect()
        frame = pd.DataFrame(results)
        if len(frame) != 6 or frame[["direction", "classifier"]].duplicated().any():
            raise ExperimentContractError("final CLIP result must contain exactly six primary rows")
        result_path = atomic_csv(self.result_root / "analysis/final_clip_results.csv", frame)
        outputs.append(result_path)
        return frame, outputs


class ExperimentReporter:
    """Writes factual execution evidence and the final integrity chain."""

    def __init__(
        self,
        repository_root: str | Path,
        artifact_root: str | Path,
        result_root: str | Path,
        config_hash: str,
        logger: ProgressLogger | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.result_root = Path(result_root).resolve()
        self.config_hash = config_hash
        self.logger = logger

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repository_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    def build(self, final_results: pd.DataFrame, prompt1: Prompt1Package) -> list[Path]:
        if len(final_results) != 6:
            raise ExperimentContractError("reporter requires six completed cells")
        report_path = self.artifact_root / "FINAL_CLIP_STABILITY_REPORT.md"
        lines = [
            "# Final CLIP Stability Execution Report",
            "",
            f"- Status: COMPLETE",
            f"- Repository commit: `{self._git_commit()}`",
            f"- Configuration SHA-256: `{self.config_hash}`",
            f"- Prompt-1 package manifest SHA-256: `{prompt1.manifest_sha256}`",
            f"- Feature universe: {FEATURE_COUNT} (`{prompt1.feature_universe_hash}`)",
            "- Directions: Stability->Stability; HomeCredit->Stability; LendingClub->Stability",
            "- Seeds: 11, 22, 33, 44, 55; checkpoint selection by minimum source validation loss",
            "- Consensus: seed-11-reference orthogonal Procrustes, five-seed normalized mean",
            "- OOT use: final evaluation only after the hashed pre-OOT freeze gate",
            "",
            "## Primary results",
            "",
            _markdown_table(final_results),
            "",
            "## Interpretation boundary",
            "",
            "AUC belongs to the Stability-trained downstream classifier, not CLIP. DEV folds are temporal diagnostics; the target-free Stability representation is frozen across row folds. Values are factual observed metrics and make no automatic superiority or significance claim.",
            "",
            "## Integrity and warnings",
            "",
            "Historical HC/LC checkpoints, source preprocessors, and source anchors were read-only and never refit. No target, OOT values, existing rankings, selector outputs, or model outputs entered CLIP representation/ranking. Thresholds were fixed from training/full-DEV predictions and not tuned on OOT.",
        ]
        _atomic_bytes(report_path, ("\n".join(lines) + "\n").encode("utf-8"))
        tracked_roots = [self.artifact_root, self.result_root]
        inventory: list[Path] = []
        for root in tracked_roots:
            inventory.extend(
                path for path in root.rglob("*")
                if path.is_file()
                and path.name not in {"sha256_manifest.csv", "final_integrity_manifest.json"}
                and "stages" not in path.relative_to(root).parts
            )
        inventory = sorted(set(path.resolve() for path in inventory), key=lambda path: path.as_posix())
        @contextmanager
        def hashing_heartbeat():
            if self.logger is None:
                yield
            else:
                with self.logger.heartbeat(
                    "[INTEGRITY] large-file hashing in progress",
                    stage="final_integrity",
                    interval=30.0,
                ):
                    yield

        log_files = [
            path for path in inventory
            if path.is_relative_to(self.artifact_root)
            and "logs" in path.relative_to(self.artifact_root).parts
        ]
        non_log_files = [path for path in inventory if path not in log_files]
        with hashing_heartbeat():
            rows = [{
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            } for path in non_log_files]
        rows.extend({
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        } for path in log_files)
        sha_path = atomic_csv(self.artifact_root / "manifests/sha256_manifest.csv", pd.DataFrame(rows))
        critical = {
            "source_authentication": self.artifact_root / "manifests/source_artifact_authentication.json",
            "stability_checkpoints": self.artifact_root / "manifests/stability_seed_checkpoints.json",
            "ranking_manifest": self.artifact_root / "manifests/ranking_manifest.json",
            "downstream_manifest": self.artifact_root / "manifests/downstream_manifest.json",
            "pre_oot_freeze": self.artifact_root / "manifests/pre_oot_freeze_manifest.json",
            "metrics": self.result_root / "analysis/final_clip_results.csv",
            "report": report_path,
            "sha256_manifest": sha_path,
        }
        integrity_path = atomic_json(self.artifact_root / "manifests/final_integrity_manifest.json", {
            "schema_version": "clip_stability_final_integrity_v1",
            "status": "COMPLETE",
            "repository_commit": self._git_commit(),
            "config_hash": self.config_hash,
            "prompt1_package_manifest_sha256": prompt1.manifest_sha256,
            "feature_universe_hash": prompt1.feature_universe_hash,
            "critical_artifact_hashes": {key: sha256_file(path) for key, path in critical.items()},
            "integrity_inventory_sha256": sha256_file(sha_path),
            "interpretation": "factual_observed_results_no_automatic_superiority_or_significance_claim",
        })
        return [report_path, sha_path, integrity_path]


def _validate_experiment_config(config: Mapping[str, Any]) -> None:
    _require_equal(config.get("schema_version"), "clip_stability_experiment_v1", "config schema")
    _require_equal(config.get("status"), "FROZEN", "config status")
    _require_equal(tuple(item["id"] for item in config["directions"]), (
        "stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability"
    ), "configured directions")
    _require_equal(tuple(config["training"]["seeds"]), REQUIRED_SEEDS, "configured seeds")
    _require_equal(config["consensus"]["reference_seed"], REFERENCE_SEED, "configured reference seed")
    _require_equal(config["prompt1_package"]["feature_universe_count"], FEATURE_COUNT, "configured feature count")
    architecture = config["representation_contract"]["architecture"]
    expected_architecture = {
        "text_input_dim": 384,
        "statistical_input_dim": 13,
        "text_hidden_dim": 64,
        "statistical_hidden_dim": 16,
        "shared_embedding_dim": 32,
        "dropout": 0.05,
        "activation": "gelu",
        "initial_temperature": 0.07,
        "trainable_temperature": False,
        "min_temperature": 0.02,
        "max_temperature": 0.5,
        "expected_parameter_count": 27488,
    }
    _require_equal(architecture, expected_architecture, "CLIP architecture")
    frozen_downstream_cells(config)


class StabilityClipExperiment:
    """Single manual orchestrator for all ten frozen stages."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.repository_root = self.config_path.parents[3]
        self.config = _read_json(self.config_path)
        _validate_experiment_config(self.config)
        self.config_hash = sha256_file(self.config_path)
        outputs = self.config["outputs"]
        self.artifact_root = _resolve(self.repository_root, outputs["artifact_root"]).resolve()
        self.result_root = _resolve(self.repository_root, outputs["result_root"]).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.logger = ProgressLogger(self.artifact_root, outputs["run_log"], outputs["progress_jsonl"])
        self.stages = StageStore(self.artifact_root, self.config_hash)

    def _stage(self, number: int, message: str) -> None:
        self.logger.emit(f"[{number}/10] {message}", stage=f"stage_{number}", event="stage_start")

    def run(self) -> None:
        experiment_manifest = atomic_json(self.artifact_root / "manifests/experiment_manifest.json", {
            "schema_version": "clip_stability_experiment_manifest_v1",
            "status": "RUNNING",
            "experiment_id": self.config["experiment_id"],
            "config_path": self.config_path.as_posix(),
            "config_hash": self.config_hash,
            "directions": [item["id"] for item in self.config["directions"]],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "isolated_artifact_root": self.artifact_root.as_posix(),
            "isolated_result_root": self.result_root.as_posix(),
        })

        self._stage(1, "Validate Prompt-1 package and methodology")
        package = Prompt1PackageValidator(self.repository_root, self.config["prompt1_package"]).validate()
        prompt1_auth = self.artifact_root / "manifests/prompt1_authentication.json"
        stage1_inputs = {"config_hash": self.config_hash, "prompt1_manifest": package.manifest_sha256}
        if self.stages.reusable("01_prompt1_authentication", stage1_inputs):
            self.logger.emit("[1/10] reusing authenticated Prompt-1 stage", stage="stage_1", event="stage_reuse")
        else:
            atomic_json(prompt1_auth, {
                "schema_version": "prompt1_clip_package_authentication_v1",
                "status": "PASS",
                "package_root": package.root.as_posix(),
                "sha256_manifest_sha256": package.manifest_sha256,
                "feature_universe_count": FEATURE_COUNT,
                "feature_universe_hash": package.feature_universe_hash,
                "target_absent": True,
                "oot_absent": True,
                "pairing_policy": "identity_equivalence_v2",
            })
            self.stages.complete("01_prompt1_authentication", stage1_inputs, [prompt1_auth])

        self._stage(2, "Authenticate corrected HC/LC source artifacts")
        resolver = HistoricalSourceArtifactResolver(self.config["representation_contract"], self.config["training"]["seeds"])
        hc = resolver.authenticate("homecredit", self.config["historical_sources"]["homecredit"])
        lc = resolver.authenticate("lendingclub", self.config["historical_sources"]["lendingclub"])
        source_auth = self.artifact_root / "manifests/source_artifact_authentication.json"
        stage2_inputs = {
            "config_hash": self.config_hash,
            "homecredit_checkpoints": canonical_hash(hc.checkpoint_hashes),
            "lendingclub_checkpoints": canonical_hash(lc.checkpoint_hashes),
        }
        if self.stages.reusable("02_source_authentication", stage2_inputs):
            self.logger.emit("[2/10] reusing authenticated HC/LC source stage", stage="stage_2", event="stage_reuse")
        else:
            atomic_json(source_auth, {
                "schema_version": "clip_source_artifact_authentication_bundle_v1",
                "status": "PASS",
                "homecredit": hc.authentication,
                "lendingclub": lc.authentication,
            })
            self.stages.complete("02_source_authentication", stage2_inputs, [source_auth])

        self._stage(3, "Train Stability CLIP - 5 seeds")
        trainer = StabilityClipTrainer(self.config, self.config_hash, self.artifact_root, self.logger, self.stages)
        results, training_summary = trainer.run(package)
        stability_checkpoints = atomic_json(self.artifact_root / "manifests/stability_seed_checkpoints.json", {
            "schema_version": "stability_clip_seed_checkpoint_bundle_v1",
            "status": "COMPLETE",
            "selection_rule": "minimum_source_validation_loss_per_seed",
            "diagnostics_not_selection": ["MRR", "Recall@1", "Recall@5", "Recall@10"],
            "seeds": {str(seed): {
                "checkpoint_path": result.checkpoint_path.as_posix(),
                "checkpoint_sha256": result.checkpoint_hash,
                "best_epoch": result.best_epoch,
                "stop_epoch": result.final_epoch,
                "best_validation_loss": result.best_validation_loss,
            } for seed, result in results.items()},
        })
        self.stages.complete(
            "03_stability_training",
            {"config_hash": self.config_hash, "prompt1_manifest": package.manifest_sha256},
            [training_summary, stability_checkpoints, *[result.checkpoint_path for result in results.values()]],
        )

        architecture = dict(self.config["representation_contract"]["architecture"])
        architecture.pop("expected_parameter_count")
        ranking_builder = TransferredRankingBuilder(self.artifact_root, package.feature_universe_hash)
        self._stage(4, "Build Stability->Stability consensus ranking")
        stability_ranking = self.artifact_root / "rankings/stability_to_stability.csv"
        stability_consensus = self.artifact_root / "representation/stability/consensus_manifest.json"
        stage4_inputs = {"config_hash": self.config_hash, "checkpoint_bundle": sha256_file(stability_checkpoints)}
        if self.stages.reusable("04_stability_ranking", stage4_inputs):
            self.logger.emit("[4/10] reusing authenticated Stability ranking", stage="stage_4", event="stage_reuse")
        else:
            stability_ranking, stability_consensus = ranking_builder.build_native(package, results, ClipModelConfig(**architecture))
            self.stages.complete("04_stability_ranking", stage4_inputs, [stability_ranking, stability_consensus])
        self._stage(5, "Build HomeCredit->Stability transfer ranking")
        hc_ranking = self.artifact_root / "rankings/homecredit_to_stability.csv"
        hc_consensus = self.artifact_root / "manifests/homecredit_to_stability_projection_consensus.json"
        stage5_inputs = {"config_hash": self.config_hash, "source_authentication": sha256_file(source_auth), "prompt1_manifest": package.manifest_sha256}
        if self.stages.reusable("05_homecredit_ranking", stage5_inputs):
            self.logger.emit("[5/10] reusing authenticated HomeCredit transfer ranking", stage="stage_5", event="stage_reuse")
        else:
            hc_ranking, hc_consensus = ranking_builder.build_transfer(package, hc, resolver)
            self.stages.complete("05_homecredit_ranking", stage5_inputs, [hc_ranking, hc_consensus])
        self._stage(6, "Build LendingClub->Stability transfer ranking")
        lc_ranking = self.artifact_root / "rankings/lendingclub_to_stability.csv"
        lc_consensus = self.artifact_root / "manifests/lendingclub_to_stability_projection_consensus.json"
        stage6_inputs = stage5_inputs
        if self.stages.reusable("06_lendingclub_ranking", stage6_inputs):
            self.logger.emit("[6/10] reusing authenticated LendingClub transfer ranking", stage="stage_6", event="stage_reuse")
        else:
            lc_ranking, lc_consensus = ranking_builder.build_transfer(package, lc, resolver)
            self.stages.complete("06_lendingclub_ranking", stage6_inputs, [lc_ranking, lc_consensus])
        ranking_manifest = atomic_json(self.artifact_root / "manifests/ranking_manifest.json", {
            "schema_version": "clip_stability_ranking_bundle_v1",
            "status": "COMPLETE",
            "feature_count_each": FEATURE_COUNT,
            "target_used": False,
            "oot_used": False,
            "rankings": {path.stem: sha256_file(path) for path in (stability_ranking, hc_ranking, lc_ranking)},
            "consensus_manifests": {path.name: sha256_file(path) for path in (stability_consensus, hc_consensus, lc_consensus)},
        })

        matrix = StabilityMatrixAccess(self.repository_root, self.config["downstream"], self.logger)
        downstream = LegacyMRMRDownstreamRunner(self.config, self.artifact_root, self.result_root, matrix, self.logger)
        self._stage(7, "Run DEV temporal downstream evaluation - 6 cells")
        stage7_inputs = {
            "config_hash": self.config_hash,
            "ranking_manifest": sha256_file(ranking_manifest),
            "matrix_manifest": self.config["downstream"]["matrix_manifest_sha256"],
            "protocol_lock": self.config["downstream"]["protocol_lock_sha256"],
        }
        expected_dev_outputs: list[Path] = [self.result_root / "analysis/dev_clip_results.csv"]
        for cell in downstream.cells:
            root = downstream._cell_root(cell)
            expected_dev_outputs.extend([
                root / "candidate_pool.csv",
                root / "dev_oof_predictions.parquet",
                root / "fold_metrics.csv",
                root / "fold_selected_features.csv",
            ])
        if self.stages.reusable("07_dev_downstream", stage7_inputs):
            self.logger.emit("[7/10] reusing authenticated DEV downstream cells", stage="stage_7", event="stage_reuse")
            dev_results = pd.read_csv(self.result_root / "analysis/dev_clip_results.csv")
            dev_outputs = expected_dev_outputs
        else:
            dev_results, dev_outputs = downstream.run_dev()
            self.stages.complete("07_dev_downstream", stage7_inputs, expected_dev_outputs)
        self._stage(8, "Fit and freeze full-DEV selectors/models")
        stage8_inputs = {
            "config_hash": self.config_hash,
            "ranking_manifest": sha256_file(ranking_manifest),
            "dev_results": sha256_file(self.result_root / "analysis/dev_clip_results.csv"),
            "protocol_lock": self.config["downstream"]["protocol_lock_sha256"],
        }
        expected_full_outputs: list[Path] = []
        for cell in downstream.cells:
            root = downstream._cell_root(cell)
            expected_full_outputs.extend([
                root / "candidate_pool.csv",
                root / "full_dev_selected_features.csv",
                root / "full_dev_score_reference.parquet",
                root / "frozen_full_dev_preprocessor.joblib",
                root / "frozen_full_dev_classifier.joblib",
                root / "full_dev_freeze_manifest.json",
            ])
        if self.stages.reusable("08_full_dev_freeze", stage8_inputs):
            self.logger.emit("[8/10] reusing authenticated full-DEV frozen cells", stage="stage_8", event="stage_reuse")
            full_outputs = expected_full_outputs
        else:
            full_outputs = downstream.fit_full_dev()
            self.stages.complete("08_full_dev_freeze", stage8_inputs, expected_full_outputs)
        downstream_manifest = atomic_json(self.artifact_root / "manifests/downstream_manifest.json", {
            "schema_version": "clip_stability_downstream_bundle_v1",
            "status": "COMPLETE",
            "cell_count": 6,
            "five_expanding_temporal_folds": True,
            "fold_fit_scope": "fold_train_only",
            "full_dev_fit_scope": "full_DEV_only",
            "candidate_pools": {"lr": 60, "catboost": 100},
            "final_k": {"lr": 20, "catboost": 40},
            "mrmr_method": RandomForestRelevanceMRMRSelector.algorithm_name,
            "dev_result_rows": len(dev_results),
            "artifact_hashes": {Path(path).resolve().as_posix(): sha256_file(path) for path in [*dev_outputs, *full_outputs]},
        })

        self._stage(9, "OOT gate + final OOT scoring - 6 cells")
        gate = PreOOTFreezeGate(self.artifact_root, self.result_root, self.config_hash)
        stage9_inputs = {
            "config_hash": self.config_hash,
            "downstream_manifest": sha256_file(downstream_manifest),
            "source_authentication": sha256_file(source_auth),
            "stability_checkpoints": sha256_file(stability_checkpoints),
        }
        expected_oot_outputs: list[Path] = [self.result_root / "analysis/final_clip_results.csv"]
        for cell in downstream.cells:
            expected_oot_outputs.extend([
                self.result_root / "predictions" / cell.direction / cell.classifier / "oot_predictions.parquet",
                downstream._cell_root(cell) / "oot_metrics.json",
            ])
        if self.stages.reusable("09_oot_final", stage9_inputs):
            token = OOTAccessToken(gate.path, sha256_file(gate.path), self.config_hash)
            gate.validate(token)
            final_results = pd.read_csv(self.result_root / "analysis/final_clip_results.csv")
            oot_outputs = expected_oot_outputs
            self.logger.emit(
                f"[9/10] reusing authenticated final OOT results pre_oot_manifest_sha256={token.manifest_sha256}",
                stage="stage_9",
                event="stage_reuse",
            )
        else:
            token = gate.create(
                prompt1_hash=package.manifest_sha256,
                feature_universe_hash=package.feature_universe_hash,
                source_authentication_path=source_auth,
                stability_checkpoint_path=stability_checkpoints,
                consensus_paths=[stability_consensus, hc_consensus, lc_consensus],
                cells=downstream.cells,
                random_seeds=REQUIRED_SEEDS,
            )
            self.logger.emit(
                f"[OOT GATE] PASS pre_oot_manifest_sha256={token.manifest_sha256} all scientific choices frozen; OOT access now permitted",
                stage="oot_gate",
                event="gate_pass",
                metrics={"pre_oot_manifest_sha256": token.manifest_sha256},
            )
            final_results, oot_outputs = FinalOOTEvaluator(
                self.config, self.result_root, matrix, gate, self.logger
            ).run(token)
            self.stages.complete("09_oot_final", stage9_inputs, [gate.path, *expected_oot_outputs])

        self._stage(10, "Build metrics, manifests, hashes, and report")
        atomic_json(experiment_manifest, {
            "schema_version": "clip_stability_experiment_manifest_v1",
            "status": "COMPLETE",
            "experiment_id": self.config["experiment_id"],
            "config_path": self.config_path.as_posix(),
            "config_hash": self.config_hash,
            "directions": [item["id"] for item in self.config["directions"]],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt1_authentication_sha256": sha256_file(prompt1_auth),
            "source_authentication_sha256": sha256_file(source_auth),
            "ranking_manifest_sha256": sha256_file(ranking_manifest),
            "downstream_manifest_sha256": sha256_file(downstream_manifest),
            "pre_oot_freeze_manifest_sha256": token.manifest_sha256,
            "final_result_sha256": sha256_file(self.result_root / "analysis/final_clip_results.csv"),
        })
        self.logger.emit("CLIP Stability experiment COMPLETE", stage="complete", event="run_complete")
        reporter = ExperimentReporter(
            self.repository_root, self.artifact_root, self.result_root, self.config_hash, self.logger
        )
        report_outputs = reporter.build(final_results, package)
        self.stages.complete(
            "10_final_reporting",
            {"config_hash": self.config_hash, "final_results": sha256_file(self.result_root / "analysis/final_clip_results.csv")},
            report_outputs,
        )


def run_stability_clip_experiment(config_path: str | Path) -> None:
    StabilityClipExperiment(config_path).run()


__all__ = [
    "AuthenticatedSource",
    "DownstreamCell",
    "ExperimentContractError",
    "ExperimentReporter",
    "FinalOOTEvaluator",
    "FiveSeedConsensusBuilder",
    "FrozenTransform",
    "HistoricalSourceArtifactResolver",
    "LegacyMRMRDownstreamRunner",
    "OOTAccessToken",
    "PreOOTFreezeGate",
    "ProgressLogger",
    "Prompt1Package",
    "Prompt1PackageValidator",
    "StabilityClipExperiment",
    "StabilityClipTrainer",
    "StabilityMatrixAccess",
    "StageStore",
    "TransferredRankingBuilder",
    "checkpoint_epoch_from_validation_losses",
    "frozen_downstream_cells",
    "identity_exclusions",
    "project_joint",
    "run_stability_clip_experiment",
    "stability_training_bundle",
]
