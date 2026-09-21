"""Step 1: Home Credit name-obfuscation ablation with both arms and a named control.

Built on the frozen helpers in ``scripts/b6_obfuscation.py`` and
``scripts/run_b6_obfuscation_ablation.py`` (record pack, seeded mapping,
scrubbing, guarded OpenAI client, strict validation, model fitting, paired
inference).  Differences from the single-arm runner:

* three conditions per partition: ``named`` control, ``A_names_removed`` and
  ``B_descriptions_removed`` (18 ranking calls, 36 refits);
* the candidate set per partition defaults to the cached run's fold-local
  candidate list (``artifacts/llm_cache``), as the TODO requires, with the full
  529 universe as an option;
* every accepted ranking, fold AUC and HO score vector is checkpointed so a
  crash never repeats an API call;
* the named rankings and fold AUCs are persisted instead of discarded.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.todo_fill import llm_cache
from scripts.todo_fill.common import (
    BUDGETS,
    FULL_DEV,
    HOMECREDIT,
    PARTITIONS,
    REPO_ROOT,
    FillError,
    Manifest,
    Skeleton,
    Timer,
    fmt,
    load_dotenv_if_present,
    log,
    read_json,
    write_json,
)
from scripts.todo_fill.data import DataContext

CONDITIONS = ("named", "A_names_removed", "B_descriptions_removed")
ARM_FILES = {"A_names_removed": "obfuscated_lists_armA.csv", "B_descriptions_removed": "obfuscated_lists_armB.csv", "named": "homecredit_named_rankings.csv"}
GUARD_CONDITION = {"named": "named", "A_names_removed": "obfuscated", "B_descriptions_removed": "named"}


def _b6():
    import scripts.b6_obfuscation as helpers
    import scripts.run_b6_obfuscation_ablation as runner

    return helpers, runner


def _blank_description_records(named_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arm B: keep every name and lineage field, blank the free-text definition."""

    helpers, _ = _b6()
    records = []
    for record in named_records:
        fields = {key: value for key, value in record.items() if key not in {"rendered_description", "description_sha256"}}
        fields["approved_definition"] = ""
        records.append(helpers._render_definition_record(fields))
    return records


def _ordered_dev(bundle) -> pd.DataFrame:
    from credit_risk_fs.experiments.lendingclub_identity import stable_chronological_order

    frame = bundle.X.copy()
    frame["__target__"] = bundle.y.to_numpy()
    frame["__time__"] = bundle.time_values.to_numpy()
    frame["__stable_id__"] = bundle.stable_row_ids.to_numpy()
    ordered = stable_chronological_order(frame, time_column="__time__", identity_column="__stable_id__")
    if len(ordered) != 99_092 or ordered["__stable_id__"].duplicated().any():
        raise FillError("ordered Home Credit DEV identity contract failed (expected 99,092 unique rows)")
    return ordered


def run_b6(*, skeleton: Skeleton, data: DataContext, work_dir: Path, out_dir: Path, manifest: Manifest, candidate_source: str = "cache") -> Skeleton:
    step = "01"
    load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY"):
        manifest.skip(step, {"dataset": HOMECREDIT}, "OPENAI_API_KEY is not set (fill .env); step 1 needs 18 gpt-4.1-mini-2025-04-14 calls")
        return skeleton
    helpers, runner = _b6()
    from sklearn.metrics import roc_auc_score
    from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit

    b6_dir = work_dir / "b6"
    (b6_dir / "rankings").mkdir(parents=True, exist_ok=True)
    (b6_dir / "fits").mkdir(parents=True, exist_ok=True)
    (b6_dir / "ho_scores").mkdir(parents=True, exist_ok=True)

    bundle = data.dev(HOMECREDIT)
    universe = list(bundle.candidate_features)
    if len(universe) != helpers.GLOBAL_FEATURE_COUNT:
        raise FillError(f"Home Credit universe has {len(universe)} features; the B6 protocol expects 529")
    mapping = helpers.build_global_feature_mapping(universe)
    reverse = helpers.reverse_feature_mapping(mapping)
    dtypes = bundle.X.dtypes.to_dict()

    # ------------------------------------------------------------ candidate sets
    candidates: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        if candidate_source == "cache":
            cached = llm_cache.ranking(HOMECREDIT, partition)
            names = [name for name in universe if name in set(cached.candidate_features)]
        else:
            names = list(universe)
        candidates[partition] = names
    write_json(b6_dir / "candidate_sets.json", {"source": candidate_source, "counts": {p: len(v) for p, v in candidates.items()}, "candidates": candidates})
    manifest.note(step, f"candidate sets per partition ({candidate_source}): {{{', '.join(f'{p}: {len(v)}' for p, v in candidates.items())}}}")

    # ------------------------------------------------------------- record packs
    packs: dict[str, dict[str, Any]] = {}
    selector = runner._selector()
    for partition, names in candidates.items():
        named_records = helpers.build_homecredit_definition_records(
            names, dtypes={name: dtypes[name] for name in names}, description_csv_path=REPO_ROOT / runner.DESCRIPTION_PATH, expected_count=len(names)
        )
        arm_a = helpers.obfuscate_definition_records(named_records, mapping)
        arm_b = _blank_description_records(named_records)
        named_ids, opaque_ids = list(names), [mapping[name] for name in names]
        named_prompt = selector.build_target_free_prompt(named_records, expected_features=named_ids)
        opaque_prompt = selector.build_target_free_prompt(arm_a, expected_features=opaque_ids)
        if opaque_prompt != helpers.scrub_text(named_prompt, mapping):
            raise FillError(f"{partition}: named and Arm A prompts differ by more than literal name replacement")
        selector.build_target_free_prompt(arm_b, expected_features=named_ids)
        packs[partition] = {
            "named": (named_records, named_ids),
            "A_names_removed": (arm_a, opaque_ids),
            "B_descriptions_removed": (arm_b, named_ids),
            "named_prompt_sha256": runner._sha256_text(named_prompt),
            "arm_a_prompt_sha256": runner._sha256_text(opaque_prompt),
        }

    # ---------------------------------------------------------------- rankings
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generator = np.random.default_rng(helpers.CALL_ORDER_SEED)
    rankings: dict[str, dict[str, list[str]]] = {condition: {} for condition in CONDITIONS}
    diagnostics: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        order = [str(item) for item in generator.permutation(np.array(CONDITIONS))]
        for sequence, condition in enumerate(order, start=1):
            checkpoint = b6_dir / "rankings" / f"{partition}__{condition}.json"
            if checkpoint.exists():
                payload = read_json(checkpoint)
            else:
                records, ids = packs[partition][condition]
                log(f"[01] ranking {partition}: condition={condition} ({sequence}/{len(CONDITIONS)}), {len(ids)} candidates")
                ranked, diagnostic = runner._rank_condition(
                    client=client, condition=GUARD_CONDITION[condition], records=records, candidate_ids=ids, original_feature_names=universe
                )
                payload = {"partition": partition, "condition": condition, "paired_sequence": sequence, "ranking": list(ranked), **diagnostic, "accepted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                write_json(checkpoint, payload)
            rankings[condition][partition] = list(payload["ranking"])
            diagnostics.append({key: value for key, value in payload.items() if key != "ranking"})
    original: dict[str, dict[str, list[str]]] = {
        "named": {p: list(v) for p, v in rankings["named"].items()},
        "A_names_removed": {p: [reverse[item] for item in v] for p, v in rankings["A_names_removed"].items()},
        "B_descriptions_removed": {p: list(v) for p, v in rankings["B_descriptions_removed"].items()},
    }

    # -------------------------------------------------------------- refits (CV)
    ordered = _ordered_dev(bundle)
    features = ordered.drop(columns=["__target__", "__time__", "__stable_id__"])
    target = ordered["__target__"].astype("int8")
    folds = list(GroupedTimeSeriesSplit(n_splits=5, gap=1).split(ordered["__time__"].to_numpy()))
    if len(folds) != 5:
        raise FillError("canonical splitter did not return five folds")
    params = runner._model_parameters(REPO_ROOT)
    np.random.seed(42)
    fold_aucs: dict[tuple[str, str], list[float]] = {}
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        partition = f"fold{fold_number}"
        for condition in CONDITIONS:
            for backbone in ("lr", "catboost"):
                k = BUDGETS[backbone]
                checkpoint = b6_dir / "fits" / f"{partition}__{condition}__{backbone}.json"
                if checkpoint.exists():
                    fold_aucs.setdefault((condition, backbone), []).append(float(read_json(checkpoint)["auc"]))
                    continue
                selected = tuple(original[condition][partition][:k])
                if len(selected) != k or len(set(selected)) != k:
                    raise FillError(f"invalid {condition}/{backbone}/{partition} ranking prefix")
                log(f"[01] fitting {partition}: condition={condition}, backbone={backbone}, K={k}")
                timer = Timer()
                _, _, _, scores = runner._fit_one(
                    backbone=backbone, params=params[backbone],
                    X_train=features.iloc[train_index].loc[:, selected], y_train=target.iloc[train_index],
                    X_validation=features.iloc[validation_index].loc[:, selected], y_validation=target.iloc[validation_index],
                )
                auc = float(roc_auc_score(target.iloc[validation_index], scores))
                write_json(checkpoint, {"partition": partition, "condition": condition, "backbone": backbone, "K": k, "auc": auc, "features": list(selected), "seconds": timer.seconds()})
                fold_aucs.setdefault((condition, backbone), []).append(auc)
                gc.collect()

    # ---------------------------------------------------- full-DEV refits + HO
    pipelines: dict[tuple[str, str], Any] = {}
    subsets = {(condition, backbone): tuple(original[condition][FULL_DEV][: BUDGETS[backbone]]) for condition in CONDITIONS for backbone in ("lr", "catboost")}
    pending_scores = [key for key in subsets if not (b6_dir / "ho_scores" / f"{key[0]}__{key[1]}.npy").exists()]
    if pending_scores:
        gate = helpers.HOAccessGate()
        gate.mark_subset_frozen()
        for key in pending_scores:
            condition, backbone = key
            log(f"[01] refitting full DEV: condition={condition}, backbone={backbone}, K={len(subsets[key])}")
            preprocessor, model, predict_proba, _ = runner._fit_one(
                backbone=backbone, params=params[backbone], X_train=features.loc[:, subsets[key]], y_train=target, X_validation=None, y_validation=None
            )
            pipelines[key] = runner.FrozenPipeline(condition=condition, backbone=backbone, selected_features=subsets[key], preprocessor=preprocessor, model=model, predict_proba=predict_proba)
        gate.mark_model_frozen()
        gate.assert_access_allowed()
        union = {feature for key in subsets for feature in subsets[key]}
        projection = [feature for feature in universe if feature in union]
        from credit_risk_fs.pipelines.common import prepare_voting_research_oot_data

        log(f"[01] loading the locked HO population projected to {len(projection)} features")
        oot = prepare_voting_research_oot_data(REPO_ROOT, dataset=HOMECREDIT, projected_candidate_features=projection, csv_chunk_rows=25_000)
        if len(oot.X) != helpers.HO_ROW_COUNT:
            raise FillError("canonical HO loader did not return exactly 120,053 rows")
        pd.DataFrame({"stable_row_id": oot.stable_row_ids.astype(str).to_numpy(), "target": oot.y.to_numpy()}).to_parquet(b6_dir / "ho_scores" / "__identity__.parquet", index=False)
        for key, pipeline in pipelines.items():
            scores = runner._score_frozen_pipeline(pipeline, oot.X)
            np.save(b6_dir / "ho_scores" / f"{key[0]}__{key[1]}.npy", scores)
        del oot
        gc.collect()
    identity = pd.read_parquet(b6_dir / "ho_scores" / "__identity__.parquet")
    ho_scores = {key: np.load(b6_dir / "ho_scores" / f"{key[0]}__{key[1]}.npy") for key in subsets}

    # ----------------------------------------------------------- inference rows
    pairwise_rows: list[dict[str, Any]] = []
    ho_auc: dict[tuple[str, str], float] = {}
    for key, scores in ho_scores.items():
        ho_auc[key] = float(roc_auc_score(identity["target"].to_numpy(dtype=int), scores))
    for backbone in ("lr", "catboost"):
        named = pd.DataFrame({"stable_row_id": identity["stable_row_id"], "target": identity["target"], "prediction_probability": ho_scores[("named", backbone)]})
        for arm in ("A_names_removed", "B_descriptions_removed"):
            treated = pd.DataFrame({"stable_row_id": identity["stable_row_id"], "target": identity["target"], "prediction_probability": ho_scores[(arm, backbone)]})
            aligned = helpers.align_paired_predictions(treated, named, production=True)
            row = helpers.build_pairwise_row(backbone=backbone, aligned=aligned, production=True)
            row["method_A"] = f"{arm} Pure LLM"
            row["method_B"] = "named Pure LLM (same run)"
            row["arm"] = arm
            pairwise_rows.append(row)

    # ------------------------------------------------------------------ outputs
    for row in skeleton.rows:
        condition, backbone = row["arm"], row["backbone"]
        if condition not in CONDITIONS or backbone not in ("lr", "catboost"):
            continue
        key = {"arm": condition, "backbone": backbone, "K": row["K"]}
        row["ho_auc"] = fmt(ho_auc[(condition, backbone)])
        manifest.cell(step, key, "ho_auc", ho_auc[(condition, backbone)], "recomputed:B6 two-arm paired run, HO 120,053 rows")
        for index, auc in enumerate(fold_aucs[(condition, backbone)], start=1):
            row[f"fold{index}_auc"] = fmt(auc)
            manifest.cell(step, key, f"fold{index}_auc", auc, "recomputed:B6 two-arm paired run")
    named_rows = [{"dataset": HOMECREDIT, "backbone": backbone, "K": BUDGETS[backbone], "ho_auc": ho_auc[("named", backbone)], **{f"fold{i}_auc": auc for i, auc in enumerate(fold_aucs[("named", backbone)], start=1)}} for backbone in ("lr", "catboost")]
    pd.DataFrame(named_rows).to_csv(out_dir / "01_obfuscation_named_control.csv", index=False, lineterminator="\n", float_format="%.17g")
    for condition, filename in ARM_FILES.items():
        rows = []
        for partition in PARTITIONS:
            for rank, item in enumerate(rankings[condition][partition], start=1):
                rows.append({"dataset": HOMECREDIT, "partition": partition, "rank": rank, "feature_id": item, "original_feature_name": reverse.get(item, item) if condition == "A_names_removed" else item})
        pd.DataFrame(rows, columns=list(helpers.RANKING_COLUMNS)).to_csv(out_dir / filename, index=False, lineterminator="\n")
    pd.DataFrame(pairwise_rows).to_csv(out_dir / "01_obfuscation_pairwise.csv", index=False, lineterminator="\n", float_format="%.17g")
    settings_line = (
        f"snapshot={helpers.MODEL_SNAPSHOT}; temperature={helpers.TEMPERATURE}; template=stability_expert_v3 target-free prompt "
        f"(source sha256 locked by scripts/run_b6_obfuscation_ablation.py preflight; named prompt sha256 per partition: "
        + ", ".join(f"{p}={packs[p]['named_prompt_sha256'][:12]}" for p in PARTITIONS)
        + f"); evidence_mode={helpers.EVIDENCE_MODE}; response_format=json_schema target_free_feature_ranking (exactly 100 distinct names, up to 3 attempts, no fallback); "
        f"shuffle_seed={helpers.MAPPING_SEED} (numpy default_rng permutation of the sorted 529 names -> F001..F529); call_order_seed={helpers.CALL_ORDER_SEED}; "
        f"candidate_sets={candidate_source} ({', '.join(f'{p}:{len(v)}' for p, v in candidates.items())}); arm B blanks approved_definition and keeps every name/lineage field; "
        f"HO rows={helpers.HO_ROW_COUNT}; refits: top-20 -> logistic regression, top-40 -> CatBoost, five folds + full DEV per arm."
    )
    (out_dir / "01_obfuscation_settings.txt").write_text(settings_line + "\n", encoding="utf-8")
    write_json(out_dir / "01_obfuscation_diagnostics.json", {"diagnostics": diagnostics, "fold_aucs": {f"{c}/{b}": v for (c, b), v in fold_aucs.items()}, "ho_auc": {f"{c}/{b}": v for (c, b), v in ho_auc.items()}, "prompt_hashes": {p: {"named": packs[p]["named_prompt_sha256"], "arm_a": packs[p]["arm_a_prompt_sha256"]} for p in PARTITIONS}})
    manifest.note(step, "wrote 01_obfuscation.csv, obfuscated_lists_armA.csv, obfuscated_lists_armB.csv, homecredit_named_rankings.csv, 01_obfuscation_named_control.csv, 01_obfuscation_pairwise.csv, 01_obfuscation_settings.txt")
    return skeleton
