from __future__ import annotations

import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from sklearn.manifold import trustworthiness
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_research_package_v2"
FIG = OUT / "figures"
DIAG = OUT / "diagnostics"
POLICY = "identity_equivalence_v2"
UMAP_SETTINGS = {
    "implementation": "umap.UMAP from umap-learn",
    "package_version": umap.__version__,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",
    "n_components": 2,
    "random_state": 42,
}
PERMUTATION_SEED = 20260625
PERMUTATIONS = 200
KNN_K = 10


def rel(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_float(value):
    return None if pd.isna(value) else float(value)


def compute_auc_ks(frame: pd.DataFrame) -> tuple[float, float]:
    if {"y_true", "y_pred_proba"}.issubset(frame.columns):
        target = frame["y_true"]
        probability = frame["y_pred_proba"]
    else:
        target = frame["target"]
        probability = frame["prediction_probability"]
    fpr, tpr, _ = roc_curve(target, probability)
    return float(roc_auc_score(target, probability)), float(np.max(tpr - fpr))


def pairwise_jaccard(path: Path) -> float:
    frame = pd.read_csv(path)
    feature_col = "feature_name" if "feature_name" in frame else "feature"
    sets = [
        set(group[feature_col].astype(str))
        for _, group in frame.groupby("fold_id", sort=True)
    ]
    values = [
        len(a & b) / len(a | b)
        for a, b in itertools.combinations(sets, 2)
    ]
    return float(np.mean(values))


def source_row(
    pipeline: str,
    model: str,
    root: Path,
    candidate_pool_size: int,
    source_dataset: str = "Home Credit",
    evaluation_dataset: str = "Home Credit",
    valid_nogueira: bool = False,
    reverse: bool = False,
) -> dict:
    summary_path = root / "results" / "experiment_summary.csv"
    summary = pd.read_csv(summary_path).iloc[0]
    prediction_path = root / "results" / "oot_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    auc, ks = compute_auc_ks(predictions)
    assert np.isclose(auc, float(summary["oot_auc"]), atol=1e-12)
    assert np.isclose(ks, float(summary["oot_ks"]), atol=1e-12)
    stability_path = root / "features" / "feature_stability_metrics.csv"
    stability = pd.read_csv(stability_path).iloc[0]
    jaccard = pairwise_jaccard(root / "features" / "fold_selected_features.csv")
    assert np.isclose(
        jaccard, float(stability["mean_pairwise_jaccard"]), atol=1e-12
    )
    split = read_json(root / "data_split_manifest.json")
    assert int(split["oot"]["row_count"]) == len(predictions)
    if reverse:
        metric_path = root / "results" / "prediction_metrics.csv"
        metric = pd.read_csv(metric_path)
        dev = metric.loc[metric["split"].eq("DEV_OOF")].iloc[0]
        oot = metric.loc[metric["split"].eq("oot")].iloc[0]
        assert int(oot["row_count"]) == len(predictions)
        psi = float(oot["score_psi"])
        dev_rows = int(dev["row_count"])
        dev_auc = float(dev["auc"])
        auc_drop = float(oot["auc_drop"])
        psi_scope = "DEV OOF reference vs OOT; DEV-OOF quantile bins"
        provenance = (
            "saved OOT and pooled DEV OOF predictions with stable SK_ID_CURR; "
            "prediction and metric manifests"
        )
        evidence = "authenticated_direct_predictions_and_manifests"
        source_metric_path = metric_path
        notes = (
            "Direct saved-prediction metric supersedes the legacy "
            "results/model_score_psi.csv scalar."
        )
    else:
        psi = float(summary["oot_model_score_psi"])
        psi_file = pd.read_csv(root / "results" / "model_score_psi.csv").iloc[0, 0]
        assert np.isclose(psi, float(psi_file), atol=1e-12)
        dev_rows = None
        dev_auc = None
        auc_drop = None
        psi_scope = (
            "final DEV-fit in-sample score distribution vs OOT; shared "
            "calculate_psi implementation"
        )
        provenance = (
            "saved 120,053-row OOT predictions; no stable borrower IDs; "
            "no pooled DEV OOF predictions"
        )
        evidence = "authenticated_oot_limited_row_identity"
        source_metric_path = summary_path
        notes = (
            "OOT AUC and KS reproduced from saved row-level predictions. "
            "DEV pooled OOF and AUC drop unavailable."
        )
    return {
        "pipeline": pipeline,
        "source_dataset": source_dataset,
        "evaluation_dataset": evaluation_dataset,
        "model": model,
        "candidate_pool_size": candidate_pool_size,
        "final_feature_count": int(summary["final_selected_feature_count"]),
        "dev_oof_rows": dev_rows,
        "dev_oof_auc": dev_auc,
        "oot_rows": len(predictions),
        "oot_auc": auc,
        "oot_ks": ks,
        "auc_drop": auc_drop,
        "score_psi": psi,
        "psi_scope": psi_scope,
        "nogueira_stability": (
            float(stability["nogueira_stability"]) if valid_nogueira else None
        ),
        "mean_pairwise_jaccard": jaccard,
        "prediction_provenance": provenance,
        "evidence_status": evidence,
        "source_metric_path": rel(source_metric_path),
        "notes": notes,
        "_prediction_path": rel(prediction_path),
        "_stability_path": rel(stability_path),
    }


def authenticate_task1() -> pd.DataFrame:
    run_roots = {
        ("Home Credit mRMR baseline", "LR"): (
            ROOT
            / "results/homecredit/lr/statistical/"
            "lr_statistical_mrmr_53a793cb32fe",
            529,
            True,
        ),
        ("Home Credit mRMR baseline", "CatBoost"): (
            ROOT
            / "results/homecredit/catboost/statistical/"
            "catboost_statistical_mrmr_3858b721e537",
            529,
            True,
        ),
        ("LLM → mRMR", "LR"): (
            ROOT
            / "results/homecredit/lr/hybrid_mrmr/"
            "lr_hybrid_llm_then_mrmr_f69e1a0cffc2",
            60,
            False,
        ),
        ("LLM → mRMR", "CatBoost"): (
            ROOT
            / "results/homecredit/catboost/hybrid_mrmr/"
            "catboost_hybrid_llm_then_mrmr_87fbcccf4952",
            100,
            False,
        ),
        ("corrected CLIP → mRMR", "LR"): (
            ROOT
            / "results/corrected_homecredit_clip/combined_pipeline/runs/"
            "homecredit_lr_corrected_clip_then_mrmr",
            60,
            False,
        ),
        ("corrected CLIP → mRMR", "CatBoost"): (
            ROOT
            / "results/corrected_homecredit_clip/combined_pipeline/runs/"
            "homecredit_catboost_corrected_clip_then_mrmr",
            100,
            False,
        ),
        ("LLM → corrected CLIP → mRMR", "LR"): (
            ROOT
            / "results/corrected_homecredit_clip/combined_pipeline/runs/"
            "homecredit_lr_llm_then_corrected_clip_then_mrmr",
            60,
            False,
        ),
        ("LLM → corrected CLIP → mRMR", "CatBoost"): (
            ROOT
            / "results/corrected_homecredit_clip/combined_pipeline/runs/"
            "homecredit_catboost_llm_then_corrected_clip_then_mrmr",
            100,
            False,
        ),
    }
    rows = []
    for (pipeline, model), (root, pool, valid_nogueira) in run_roots.items():
        rows.append(
            source_row(
                pipeline,
                model,
                root,
                pool,
                valid_nogueira=valid_nogueira,
            )
        )
    reverse_roots = {
        "LR": ROOT
        / "results/corrected_lendingclub_to_homecredit_transfer/downstream/"
        "logistic_regression",
        "CatBoost": ROOT
        / "results/corrected_lendingclub_to_homecredit_transfer/downstream/catboost",
    }
    for model, root in reverse_roots.items():
        rows.append(
            source_row(
                "LendingClub corrected CLIP → Home Credit",
                model,
                root,
                60 if model == "LR" else 100,
                source_dataset="LendingClub v2",
                evaluation_dataset="Home Credit",
                valid_nogueira=True,
                reverse=True,
            )
        )
    table = pd.DataFrame(rows)

    combined_manifest = read_json(
        ROOT
        / "results/corrected_homecredit_clip/combined_pipeline/"
        "combined_pipeline_manifest.json"
    )
    assert combined_manifest["pairing_policy_version"] == POLICY
    assert combined_manifest["candidate_pool_sizes"] == {"lr": 60, "catboost": 100}

    reusable = pd.read_csv(ROOT / "results/research_summary/reusable_metrics.csv")
    mapping = {
        "Home Credit mRMR baseline": "mrmr",
        "LLM → mRMR": "llm_then_mrmr",
        "corrected CLIP → mRMR": "corrected_clip_then_mrmr",
        "LLM → corrected CLIP → mRMR": "llm_then_corrected_clip_then_mrmr",
        "LendingClub corrected CLIP → Home Credit": (
            "lendingclub_clip_to_homecredit_mrmr"
        ),
    }
    for row in table.itertuples():
        candidates = reusable[
            reusable["dataset_name"].eq("homecredit")
            & reusable["model"].eq("lr" if row.model == "LR" else "catboost")
            & reusable["selector"].eq(mapping[row.pipeline])
        ]
        assert len(candidates) == 1
        registry = candidates.iloc[0]
        assert np.isclose(row.oot_auc, registry["oot_auc"], atol=1e-12)
        assert np.isclose(row.oot_ks, registry["oot_ks"], atol=1e-12)
        assert np.isclose(row.score_psi, registry["model_score_psi"], atol=1e-12)
    return table


def incremental_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["LR", "CatBoost"]:
        llm = table[
            table["pipeline"].eq("LLM → mRMR") & table["model"].eq(model)
        ].iloc[0]
        combined = table[
            table["pipeline"].eq("LLM → corrected CLIP → mRMR")
            & table["model"].eq(model)
        ].iloc[0]
        if model == "LR":
            category = "available evidence is inconclusive"
            conclusion = (
                "No predictive gain; feature-selection stability improved, "
                "while score PSI was slightly higher (worse)."
            )
        else:
            category = "corrected CLIP improved stability but not prediction"
            conclusion = (
                "The AUC change was tiny and KS decreased; score PSI and "
                "feature-selection stability improved."
            )
        rows.append(
            {
                "model": model,
                "llm_only_oot_auc": llm["oot_auc"],
                "combined_oot_auc": combined["oot_auc"],
                "delta_oot_auc_combined_minus_llm": (
                    combined["oot_auc"] - llm["oot_auc"]
                ),
                "auc_direction": (
                    "higher" if combined["oot_auc"] > llm["oot_auc"] else "lower"
                ),
                "llm_only_oot_ks": llm["oot_ks"],
                "combined_oot_ks": combined["oot_ks"],
                "delta_oot_ks_combined_minus_llm": (
                    combined["oot_ks"] - llm["oot_ks"]
                ),
                "ks_direction": (
                    "higher" if combined["oot_ks"] > llm["oot_ks"] else "lower"
                ),
                "llm_only_score_psi": llm["score_psi"],
                "combined_score_psi": combined["score_psi"],
                "delta_score_psi_combined_minus_llm": (
                    combined["score_psi"] - llm["score_psi"]
                ),
                "psi_direction": (
                    "higher_worse"
                    if combined["score_psi"] > llm["score_psi"]
                    else "lower_better"
                ),
                "psi_comparability": (
                    "comparable: same Home Credit DEV/OOT windows and shared "
                    "final-DEV-fit-score vs OOT calculate_psi implementation"
                ),
                "llm_only_mean_pairwise_jaccard": llm["mean_pairwise_jaccard"],
                "combined_mean_pairwise_jaccard": combined[
                    "mean_pairwise_jaccard"
                ],
                "delta_jaccard_combined_minus_llm": (
                    combined["mean_pairwise_jaccard"]
                    - llm["mean_pairwise_jaccard"]
                ),
                "stability_metric": (
                    "mean pairwise Jaccard recomputed from saved fold sets; "
                    "saved Nogueira/Kuncheva omitted because they used N=529 "
                    "instead of the authenticated 60/100 candidate pools"
                ),
                "required_category": category,
                "incremental_value_conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"joint_\d{4}", column)
    ]
    assert len(columns) == 32
    values = frame[columns].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    return values


def embedding_diagnostics(
    frame: pd.DataFrame,
    values: np.ndarray,
    direction: str,
    source_representation: str,
    projected_dataset: str,
    source_embedding_path: Path,
    source_manifest_path: Path,
    excluded_count: int,
    exclusion_reasons: str,
    taxonomy: str,
    comparison_limitations: str,
    coord_name: str,
    manifest_name: str,
    figure_name: str,
    title: str,
    authentication: dict,
) -> dict:
    assert frame["feature_name"].is_unique
    assert frame["semantic_group"].notna().all()
    labels = frame["semantic_group"].astype(str).to_numpy()

    # This duplicates the validated Task 3 operational procedure exactly.
    neighbours = (
        NearestNeighbors(n_neighbors=11, metric="cosine")
        .fit(values)
        .kneighbors(return_distance=False)[:, 1:]
    )
    assert neighbours.shape == (len(frame), KNN_K)
    observed = float(np.mean(labels[neighbours] == labels[:, None]))
    rng = np.random.default_rng(PERMUTATION_SEED)
    shuffled = []
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(labels)
        shuffled.append(float(np.mean(permuted[neighbours] == permuted[:, None])))

    reducer = umap.UMAP(
        n_neighbors=UMAP_SETTINGS["n_neighbors"],
        min_dist=UMAP_SETTINGS["min_dist"],
        metric=UMAP_SETTINGS["metric"],
        n_components=UMAP_SETTINGS["n_components"],
        random_state=UMAP_SETTINGS["random_state"],
    )
    coordinates = reducer.fit_transform(values)
    trust = float(
        trustworthiness(
            values,
            coordinates,
            n_neighbors=KNN_K,
            metric="cosine",
        )
    )
    coord = frame[["feature_name", "semantic_group"]].copy()
    if "feature_id" in frame:
        coord.insert(0, "feature_id", frame["feature_id"].values)
    coord["umap_1"] = coordinates[:, 0]
    coord["umap_2"] = coordinates[:, 1]
    coord_path = DIAG / coord_name
    coord.to_csv(coord_path, index=False)

    groups = sorted(coord["semantic_group"].unique())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(groups), endpoint=False))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    fig, ax = plt.subplots(figsize=(13, 9))
    for index, group in enumerate(groups):
        mask = coord["semantic_group"].eq(group).to_numpy()
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=30,
            alpha=0.78,
            color=colors[index],
            marker=markers[index % len(markers)],
            edgecolors="none",
            label=group,
        )
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.15)
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[index % len(markers)],
            color="none",
            markerfacecolor=colors[index],
            markeredgecolor="none",
            markersize=7,
            label=group,
        )
        for index, group in enumerate(groups)
    ]
    ax.legend(
        handles=handles,
        title=f"Semantic group ({len(groups)})",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    subtitle = (
        f"n={len(frame)} {projected_dataset} features; semantic-group colors; "
        f"new coordinates from saved embeddings\n"
        f"Source representation: {source_representation}; "
        f"UMAP cosine, neighbours=15, min_dist=0.1, seed=42"
    )
    fig.text(0.5, 0.015, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 0.82, 1))
    fig.savefig(FIG / figure_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = {
        "direction": direction,
        "source_representation": source_representation,
        "projected_dataset": projected_dataset,
        "valid_embedding_count": len(frame),
        "semantic_group_count": len(groups),
        "knn_k": KNN_K,
        "cosine_knn_purity": observed,
        "shuffle_permutations": PERMUTATIONS,
        "shuffled_mean": float(np.mean(shuffled)),
        "shuffled_p95": float(np.quantile(shuffled, 0.95)),
        "shuffled_p975": float(np.quantile(shuffled, 0.975)),
        "umap_trustworthiness": trust,
        "excluded_feature_count": excluded_count,
        "main_exclusion_reasons": exclusion_reasons,
        "semantic_taxonomy": taxonomy,
        "comparison_limitations": comparison_limitations,
        "source_embedding_path": rel(source_embedding_path),
        "source_manifest_path": rel(source_manifest_path),
    }
    manifest = {
        "status": "authenticated_report_diagnostic",
        "direction": direction,
        "coordinates_status": "newly calculated from immutable saved embeddings",
        "source_embedding_path": rel(source_embedding_path),
        "source_embedding_sha256": sha256(source_embedding_path),
        "source_manifest_path": rel(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "pairing_policy_version": POLICY,
        "embedding_rows": len(frame),
        "embedding_dimensions": values.shape[1],
        "feature_identity_unique": bool(frame["feature_name"].is_unique),
        "semantic_labels_complete": bool(frame["semantic_group"].notna().all()),
        "semantic_group_count": len(groups),
        "umap_settings": UMAP_SETTINGS,
        "knn_procedure": {
            "validated_task3_replication": True,
            "metric": "cosine",
            "k": KNN_K,
            "implementation": (
                "NearestNeighbors(n_neighbors=11).kneighbors"
                "(return_distance=False)[:,1:]"
            ),
        },
        "shuffle_procedure": {
            "seed": PERMUTATION_SEED,
            "permutations": PERMUTATIONS,
            "labels_permuted_with_embeddings_fixed": True,
        },
        "diagnostics": result,
        "authentication": authentication,
        "coordinate_path": rel(coord_path),
        "coordinate_sha256": sha256(coord_path),
        "figure_path": rel(FIG / figure_name),
        "figure_sha256": sha256(FIG / figure_name),
        "interpretation_limit": (
            "UMAP axes have no direct substantive interpretation; separate "
            "UMAP layouts are not geometrically comparable and do not establish "
            "downstream predictive utility."
        ),
    }
    (DIAG / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def authenticate_embeddings() -> pd.DataFrame:
    forward_path = (
        ROOT
        / "results/corrected_homecredit_clip/training/"
        "lendingclub_v2_joint_embeddings.parquet"
    )
    forward_manifest = (
        ROOT / "results/corrected_homecredit_clip/training/training_manifest.json"
    )
    forward = pd.read_parquet(forward_path)
    assert len(forward) == 576
    assert forward["feature_name"].is_unique
    assert forward["dataset"].eq("lendingclub_v2").all()
    assert forward["split"].eq("external_validation").all()
    assert forward["pairing_policy_version"].eq(POLICY).all()
    assert forward["semantic_group"].notna().all()
    forward_training = read_json(forward_manifest)
    forward_summary = read_json(
        ROOT
        / "results/corrected_homecredit_clip/training/training_summary.json"
    )
    assert forward_training["counts"]["lendingclub_v2_external_pairs"] == 576
    assert forward_training["external_validation_dataset"] == "lendingclub_v2"
    assert forward_summary["lendingclub_v2_used_for_training"] is False
    checkpoint = (
        ROOT
        / "results/corrected_homecredit_clip/training/seeds/seed_55/"
        "best_checkpoint.pt"
    )
    assert sha256(checkpoint) == forward_summary["selected_checkpoint_hash"]
    assert forward["checkpoint_hash"].eq(sha256(checkpoint)).all()
    forward_diag = embedding_diagnostics(
        forward,
        embedding_matrix(forward),
        "Home Credit → LendingClub v2",
        "corrected Home Credit-trained CLIP (selected seed 55)",
        "LendingClub v2",
        forward_path,
        forward_manifest,
        0,
        "none within the authenticated 576-row external-pair universe",
        "LendingClub v2 semantic-group taxonomy (17 groups)",
        (
            "The LendingClub and Home Credit taxonomies differ materially; "
            "purity values must not be ranked as if labels were identical."
        ),
        "forward_homecredit_to_lendingclub_umap_coordinates.csv",
        "forward_homecredit_to_lendingclub_umap_manifest.json",
        "figure_3_forward_homecredit_to_lendingclub_umap.png",
        "Home Credit-trained corrected CLIP projected to LendingClub v2",
        {
            "selected_checkpoint_path": rel(checkpoint),
            "selected_checkpoint_sha256": sha256(checkpoint),
            "external_dataset_used_for_training": False,
            "external_dataset_used_for_model_selection": False,
            "external_pair_count": 576,
            "feature_to_embedding_mapping": "one unique feature_name per row",
        },
    )

    reverse_path = (
        ROOT
        / "results/corrected_lendingclub_to_homecredit_transfer/"
        "reverse_projection/homecredit_reverse_embeddings.parquet"
    )
    reverse_manifest = (
        ROOT
        / "results/corrected_lendingclub_to_homecredit_transfer/"
        "reverse_projection/reverse_projection_manifest.json"
    )
    reverse = pd.read_parquet(reverse_path)
    manifest = read_json(reverse_manifest)
    assert len(reverse) == manifest["projected_feature_count"] == 436
    assert reverse["feature_id"].is_unique
    assert reverse["feature_name"].is_unique
    assert reverse["source_dataset"].eq("lendingclub_v2").all()
    assert reverse["external_dataset"].eq("homecredit").all()
    assert reverse["pairing_policy_version"].eq(POLICY).all()
    universe = pd.read_csv(
        ROOT
        / "results/corrected_homecredit_clip/feature_universe/"
        "feature_universe_reconciliation.csv"
    )[["feature_id", "semantic_group"]]
    reverse = reverse.merge(
        universe, on="feature_id", how="left", validate="one_to_one"
    )
    assert reverse["semantic_group"].notna().all()
    reverse_ids = set(reverse["feature_id"])
    seed_hashes = {}
    for seed in [11, 22, 33, 44, 55]:
        seed_path = (
            ROOT
            / "results/corrected_lendingclub_to_homecredit_transfer/"
            f"reverse_projection/seed_{seed}_homecredit_reverse_embeddings.parquet"
        )
        seed_frame = pd.read_parquet(seed_path)
        assert set(seed_frame["feature_id"]) == reverse_ids
        seed_hashes[str(seed)] = sha256(seed_path)
    reconciliation = pd.read_csv(
        ROOT
        / "results/corrected_lendingclub_to_homecredit_transfer/"
        "reverse_projection/homecredit_reverse_feature_reconciliation.csv"
    )
    excluded = reconciliation.loc[~reconciliation["compatible"]]
    assert len(excluded) == 95
    reverse_diag = embedding_diagnostics(
        reverse,
        embedding_matrix(reverse),
        "LendingClub v2 → Home Credit",
        "five-seed corrected LendingClub v2-trained CLIP consensus",
        "Home Credit",
        reverse_path,
        reverse_manifest,
        len(excluded),
        "missing_semantic_text_embedding: 95",
        "Home Credit semantic-group taxonomy (11 represented groups)",
        (
            "The Home Credit and LendingClub taxonomies differ materially; "
            "purity values must not be ranked as if labels were identical."
        ),
        "reverse_lendingclub_to_homecredit_umap_coordinates.csv",
        "reverse_lendingclub_to_homecredit_umap_manifest.json",
        "figure_4_reverse_lendingclub_to_homecredit_umap.png",
        "LendingClub v2-trained corrected CLIP projected to Home Credit",
        {
            "seed_embedding_paths_and_sha256": {
                rel(
                    ROOT
                    / "results/corrected_lendingclub_to_homecredit_transfer/"
                    f"reverse_projection/seed_{seed}_homecredit_reverse_embeddings.parquet"
                ): seed_hashes[str(seed)]
                for seed in [11, 22, 33, 44, 55]
            },
            "reference_seed": manifest["reference_seed"],
            "alignment_method": manifest["alignment_method"],
            "embedding_aggregation": manifest["embedding_aggregation"],
            "external_refit": manifest["external_refit"],
            "feature_reconciliation_rows": len(reconciliation),
            "eligible_rows": int(reconciliation["compatible"].sum()),
            "excluded_rows": len(excluded),
        },
    )
    return pd.DataFrame([forward_diag, reverse_diag])


def plot_task1(table: pd.DataFrame, incremental: pd.DataFrame) -> None:
    order = [
        "Home Credit mRMR baseline",
        "LLM → mRMR",
        "corrected CLIP → mRMR",
        "LLM → corrected CLIP → mRMR",
        "LendingClub corrected CLIP → Home Credit",
    ]
    labels = [
        "mRMR",
        "LLM → mRMR",
        "corrected CLIP → mRMR",
        "LLM → corrected CLIP → mRMR",
        "LC → HC transfer",
    ]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 7))
    for offset, model, color in [
        (-width / 2, "LR", "#2f6b9a"),
        (width / 2, "CatBoost", "#d8782d"),
    ]:
        values = [
            table.loc[
                table["pipeline"].eq(pipeline) & table["model"].eq(model),
                "oot_auc",
            ].iloc[0]
            for pipeline in order
        ]
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Authenticated Home Credit pipeline OOT AUC")
    ax.set_ylabel("OOT ROC AUC")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIG / "figure_1_homecredit_pipeline_oot_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    methods = ["LLM → mRMR", "LLM → corrected CLIP → mRMR"]
    short = ["LLM → mRMR", "LLM → CLIP → mRMR"]
    metrics = [
        ("oot_auc", "OOT ROC AUC", None),
        ("oot_ks", "OOT KS", None),
        ("score_psi", "Score PSI (lower is steadier)", None),
        ("mean_pairwise_jaccard", "Mean pairwise Jaccard", None),
    ]
    for ax, (metric, title, _) in zip(axes.flat, metrics):
        xx = np.arange(2)
        for offset, model, color in [
            (-width / 2, "LR", "#2f6b9a"),
            (width / 2, "CatBoost", "#d8782d"),
        ]:
            vals = [
                table.loc[
                    table["pipeline"].eq(method) & table["model"].eq(model),
                    metric,
                ].iloc[0]
                for method in methods
            ]
            bars = ax.bar(xx + offset, vals, width, label=model, color=color)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(xx, short, rotation=12, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Incremental value of corrected CLIP after LLM",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        (
            "PSI is comparable only within these Task 1 pairs. Jaccard is "
            "recomputed from saved fold-selected sets; invalid universe-based "
            "Nogueira/Kuncheva values are omitted."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(
        FIG / "figure_2_task1_incremental_value.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    reverse = table[
        table["pipeline"].eq("LendingClub corrected CLIP → Home Credit")
    ].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "oot_auc", "OOT ROC AUC"),
        (axes[1], "score_psi", "Score PSI"),
        (axes[2], "mean_pairwise_jaccard", "Selection Jaccard"),
    ]:
        vals = [reverse.loc["LR", metric], reverse.loc["CatBoost", metric]]
        bars = ax.bar(["LR", "CatBoost"], vals, color=["#2f6b9a", "#d8782d"])
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("LendingClub v2 → Home Credit performance and stability")
    fig.text(
        0.5,
        0.01,
        (
            "PSI uses saved pooled DEV OOF probabilities as reference; "
            "Jaccard uses the authenticated 60/100 candidate universes."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(
        FIG / "figure_5_reverse_transfer_performance_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def report_markdown(
    table: pd.DataFrame,
    incremental: pd.DataFrame,
    directional: pd.DataFrame,
) -> str:
    lookup = table.set_index(["pipeline", "model"])
    inc = incremental.set_index("model")
    forward = directional.iloc[0]
    reverse = directional.iloc[1]
    lr_llm = lookup.loc[("LLM → mRMR", "LR")]
    cb_llm = lookup.loc[("LLM → mRMR", "CatBoost")]
    lr_comb = lookup.loc[("LLM → corrected CLIP → mRMR", "LR")]
    cb_comb = lookup.loc[("LLM → corrected CLIP → mRMR", "CatBoost")]
    lr_clip = lookup.loc[("corrected CLIP → mRMR", "LR")]
    cb_clip = lookup.loc[("corrected CLIP → mRMR", "CatBoost")]
    lr_base = lookup.loc[("Home Credit mRMR baseline", "LR")]
    cb_base = lookup.loc[("Home Credit mRMR baseline", "CatBoost")]
    lr_rev = lookup.loc[("LendingClub corrected CLIP → Home Credit", "LR")]
    cb_rev = lookup.loc[("LendingClub corrected CLIP → Home Credit", "CatBoost")]
    return f"""
# Corrected CLIP Credit-Feature Research: Complete Task 1 Comparison and Directional Transfer Evidence

## Abstract

This revision closes two reporting gaps in the original final package without rerunning any scientific stage. First, it authenticates the standalone LLM → mRMR pipeline and compares it directly with corrected CLIP → mRMR and LLM → corrected CLIP → mRMR for Home Credit Logistic Regression (LR) and CatBoost. Second, it separates the two representation-transfer directions: a Home Credit-trained corrected CLIP representation projected onto 576 LendingClub v2 features, and a five-seed LendingClub v2-trained corrected CLIP consensus projected onto 436 Home Credit features. All predictive values come from saved row-level OOT predictions or their authenticated metric artifacts; all embedding diagnostics are new read-only calculations from saved matrices.

The central incremental result is mixed. For LR, adding corrected CLIP after LLM changed OOT AUC from {fmt(lr_llm.oot_auc)} to {fmt(lr_comb.oot_auc)} (Δ {inc.loc["LR", "delta_oot_auc_combined_minus_llm"]:+.6f}) and KS from {fmt(lr_llm.oot_ks)} to {fmt(lr_comb.oot_ks)} (Δ {inc.loc["LR", "delta_oot_ks_combined_minus_llm"]:+.6f}); this is not a predictive improvement. Pairwise feature-selection Jaccard rose from {fmt(lr_llm.mean_pairwise_jaccard)} to {fmt(lr_comb.mean_pairwise_jaccard)}, but score PSI increased slightly from {fmt(lr_llm.score_psi)} to {fmt(lr_comb.score_psi)}. For CatBoost, AUC rose by only {inc.loc["CatBoost", "delta_oot_auc_combined_minus_llm"]:+.6f}, while KS fell by {abs(inc.loc["CatBoost", "delta_oot_ks_combined_minus_llm"]):.6f}; the changes do not establish meaningful predictive improvement. CatBoost stability was clearer: PSI fell from {fmt(cb_llm.score_psi)} to {fmt(cb_comb.score_psi)} and Jaccard rose from {fmt(cb_llm.mean_pairwise_jaccard)} to {fmt(cb_comb.mean_pairwise_jaccard)}.

Both directional embedding spaces show local semantic structure far above shuffled-label references, but their taxonomies differ and their UMAP coordinates are not directly comparable. Reverse downstream prediction was materially weaker than Home Credit-trained pipelines. Representation evidence is therefore bidirectional; competitive predictive evidence is not.

## 1. Research questions

The revision answers eight linked but distinct questions. How well does LLM → mRMR perform? How well does corrected CLIP → mRMR perform? How well does their combination perform? Does corrected CLIP add OOT AUC, OOT KS, score stability, or feature-selection stability after the LLM? Does that answer depend on the downstream learner? What structure is visible in each transfer direction? Does either direction support competitive downstream predictive transfer? Finally, is the evidence for bidirectional representation structure stronger than the evidence for bidirectional prediction?

These questions must remain separate. An embedding can preserve semantic neighbourhoods without preserving outcome discrimination. A candidate selector can become more stable while AUC remains unchanged. A low PSI can indicate a steady score distribution even when the score ranks risk poorly. A UMAP can faithfully display local geometry without proving that any displayed cluster has predictive value. The report therefore treats OOT discrimination, score distribution stability, feature-set reproducibility, and embedding structure as different evidence classes.

The source hierarchy follows the audit specification. Saved row-level predictions receive priority, followed by prediction and metric manifests, completed stage manifests, feature-selection manifests, embedding manifests, registries, and finally the v1 report as a writing reference. The deleted 182-run matrix was not restored or used. Old-policy CLIP checkpoints and their dependent results were excluded.

## 2. Data and validation design

Home Credit combines borrower application data with bureau, previous-application, installment, point-of-sale, and credit-card aggregates. LendingClub v2 represents a different lending process and feature vocabulary, including revolving utilization, inquiry, delinquency, credit-depth, FICO, mortgage, exposure, and loan-term groups. The domains overlap conceptually but not by row identity or feature taxonomy.

The Home Credit comparison uses the same declared DEV interval, days [-600, -240), and OOT interval, days [-240, 0). Every authenticated Home Credit OOT prediction file contains 120,053 rows. Task 1 files retain targets and probabilities but not stable borrower identifiers. Their OOT AUC and KS can be reproduced exactly, but dataframe row position is not treated as identity, cross-method paired testing is not attempted, and historical mean fold AUC is not relabelled as pooled OOF AUC. DEV OOF rows, pooled OOF AUC, and AUC drop are therefore blank for the four Home Credit-trained pipelines.

The reverse-transfer workflow has stronger row provenance. Its prediction files retain `SK_ID_CURR`; 82,647 pooled DEV OOF rows reconcile to validation folds, and 120,053 OOT rows are authenticated through prediction and metric manifests. Its AUC drop is DEV OOF AUC minus OOT AUC. LR has DEV OOF AUC {fmt(lr_rev.dev_oof_auc)} and OOT AUC {fmt(lr_rev.oot_auc)}, giving {lr_rev.auc_drop:+.6f}. CatBoost has DEV OOF AUC {fmt(cb_rev.dev_oof_auc)} and OOT AUC {fmt(cb_rev.oot_auc)}, giving {cb_rev.auc_drop:+.6f}.

Task 1 PSI is calculated by the shared historical pipeline from final DEV-fit in-sample scores versus OOT scores. Because the standalone and combined methods use the same Home Credit windows and implementation, PSI is methodologically comparable within each Task 1 model pair. Reverse-transfer PSI instead uses saved pooled DEV OOF probabilities as reference, frozen DEV-quantile bins, and OOT probabilities as comparison. Task 1 and reverse PSI are retained but not compared numerically as if their reference scopes were identical.

## 3. Methodology

### LLM semantic screening

The LLM stage is a domain-facing screen. It uses names, descriptions, and semantic context to prioritize plausible credit-risk features before supervised selection. It does not learn borrower-level predictions and does not authenticate temporal performance. In the saved LLM → mRMR workflow, each LR fold and final-DEV selection uses a 60-feature LLM candidate list; CatBoost uses 100. The trace contains six scopes per model—five folds and final DEV—and confirms these widths.

Semantic screening can remove obviously irrelevant or redundant concepts early, reducing the supervised search space. It can also be imperfect because a feature name may conceal useful transformations or because semantically plausible variables may carry little incremental target signal. Its value must therefore be judged downstream and cannot be inferred from prompt quality alone.

### Corrected CLIP representation filtering

Corrected CLIP aligns text descriptions with target-free statistical descriptors. The corrected negative policy is `identity_equivalence_v2`: only verified identity-equivalent pairs are masked as non-negatives. Same source table, broad family, text similarity, or statistical similarity are diagnostic relations rather than broad exclusions. Five corrected seeds are available for both representation programmes, and old invalid checkpoints were rejected.

For the Home Credit-trained representation, LendingClub v2 supplied external-validation pairs only. It was not used for training or checkpoint selection. The selected seed-55 checkpoint projects each of 576 LendingClub features into a saved 32-dimensional joint space. For reverse transfer, five LendingClub-trained seed spaces were aligned to seed 11 by orthogonal Procrustes transformation, averaged after normalization, and normalized again. Frozen projection produced a 436-row Home Credit consensus matrix without external refitting.

### mRMR supervised selection

mRMR supplies the target-aware stage. It selects variables that are relevant to the Home Credit outcome while penalizing redundancy. It operates inside DEV only and reduces 60 candidates to 20 for LR or 100 candidates to 40 for CatBoost. The baseline mRMR rows begin from all 529 authenticated Home Credit candidates.

### Why the three roles are complementary

The LLM answers whether a feature is semantically plausible; corrected CLIP answers whether semantic and target-free statistical views form a coherent representation; mRMR answers whether candidates add supervised relevance without excessive duplication. These are complementary filters, not interchangeable estimators. The incremental experiment is necessary precisely because conceptual complementarity does not guarantee an empirical gain.

## 4. Complete Home Credit pipeline comparison

The complete table is [final_results_tables.csv](final_results_tables.csv). Figure 1 uses only authenticated OOT AUC values and begins at zero.

![Authenticated Home Credit pipeline OOT AUC](figures/figure_1_homecredit_pipeline_oot_auc.png)

*Figure 1. OOT ROC AUC for the Home Credit mRMR baseline, LLM → mRMR, corrected CLIP → mRMR, LLM → corrected CLIP → mRMR, and LendingClub → Home Credit reverse transfer. Bars are grouped by LR and CatBoost. Sources: saved OOT prediction paths recorded in `final_results_tables.csv`; the y-axis begins at zero.*

The full mRMR baseline reached LR AUC {fmt(lr_base.oot_auc)}, KS {fmt(lr_base.oot_ks)}, and Jaccard {fmt(lr_base.mean_pairwise_jaccard)} with 20 of 529 features. CatBoost reached AUC {fmt(cb_base.oot_auc)}, KS {fmt(cb_base.oot_ks)}, and Jaccard {fmt(cb_base.mean_pairwise_jaccard)} with 40 features. These were the strongest AUC values among the required Home Credit-trained rows, though no paired significance claim is possible.

LLM → mRMR reached LR AUC {fmt(lr_llm.oot_auc)}, KS {fmt(lr_llm.oot_ks)}, score PSI {fmt(lr_llm.score_psi)}, and Jaccard {fmt(lr_llm.mean_pairwise_jaccard)}. CatBoost reached AUC {fmt(cb_llm.oot_auc)}, KS {fmt(cb_llm.oot_ks)}, PSI {fmt(cb_llm.score_psi)}, and Jaccard {fmt(cb_llm.mean_pairwise_jaccard)}. The candidate pools were 60 and 100, not 529.

Corrected CLIP → mRMR reached LR AUC {fmt(lr_clip.oot_auc)}, KS {fmt(lr_clip.oot_ks)}, PSI {fmt(lr_clip.score_psi)}, and Jaccard {fmt(lr_clip.mean_pairwise_jaccard)}. CatBoost reached AUC {fmt(cb_clip.oot_auc)}, KS {fmt(cb_clip.oot_ks)}, PSI {fmt(cb_clip.score_psi)}, and Jaccard {fmt(cb_clip.mean_pairwise_jaccard)}. Corrected CLIP alone therefore produced a stable selector but did not exceed the full mRMR baseline.

The combined LLM → corrected CLIP → mRMR pipeline reached LR AUC {fmt(lr_comb.oot_auc)}, KS {fmt(lr_comb.oot_ks)}, PSI {fmt(lr_comb.score_psi)}, and Jaccard {fmt(lr_comb.mean_pairwise_jaccard)}. CatBoost reached AUC {fmt(cb_comb.oot_auc)}, KS {fmt(cb_comb.oot_ks)}, PSI {fmt(cb_comb.score_psi)}, and Jaccard {fmt(cb_comb.mean_pairwise_jaccard)}.

The saved Task 1 Nogueira and Kuncheva values are not reported. Their files specify a 529-feature universe, but the authenticated selection manifests show 60 and 100 candidates. Those measures depend on the universe size. Mean pairwise Jaccard was independently recomputed from the five saved fold-selected sets and matched the saved Jaccard scalar exactly, so it is the valid stability comparator.

## 5. Incremental value of corrected CLIP after LLM

[task1_incremental_comparison.csv](task1_incremental_comparison.csv) records each input value, difference, direction, comparability decision, and required categorical conclusion.

![Incremental value of corrected CLIP after LLM](figures/figure_2_task1_incremental_value.png)

*Figure 2. Direct LLM → mRMR versus LLM → corrected CLIP → mRMR comparison. AUC, KS, comparable within-Task-1 PSI, and independently verified pairwise Jaccard are shown separately. Sources are the saved OOT metrics and fold-selected feature sets listed in `final_results_tables.csv`.*

### LR result

For LR, combined minus LLM-only AUC is {inc.loc["LR", "delta_oot_auc_combined_minus_llm"]:+.6f}; AUC became slightly lower. KS changed by {inc.loc["LR", "delta_oot_ks_combined_minus_llm"]:+.6f}, a tiny increase. These changes point in opposite directions and are too small, without paired testing, to support a meaningful predictive improvement. The correct predictive conclusion is that LR did not benefit.

Score PSI changed by {inc.loc["LR", "delta_score_psi_combined_minus_llm"]:+.6f}; because lower is preferred, score stability was slightly worse. Pairwise Jaccard changed by {inc.loc["LR", "delta_jaccard_combined_minus_llm"]:+.6f}, a large increase in fold-to-fold feature overlap. LR therefore shows improved feature-selection stability but not improved score stability. Under the required categories, the overall available evidence is inconclusive rather than uniformly positive.

### CatBoost result

For CatBoost, combined minus LLM-only AUC is {inc.loc["CatBoost", "delta_oot_auc_combined_minus_llm"]:+.6f}. KS changed by {inc.loc["CatBoost", "delta_oot_ks_combined_minus_llm"]:+.6f}. The AUC increase is under one-thousandth and KS moved in the opposite direction. With no stable row IDs for a paired test, this is not evidence of a meaningful predictive improvement or equivalence.

CatBoost score PSI changed by {inc.loc["CatBoost", "delta_score_psi_combined_minus_llm"]:+.6f}, a lower and therefore better value. Jaccard changed by {inc.loc["CatBoost", "delta_jaccard_combined_minus_llm"]:+.6f}. CatBoost consequently supports improved score and feature-selection stability, but not clear predictive improvement. The required category is “corrected CLIP improved stability but not prediction.”

Across models, corrected CLIP after LLM primarily regularized which features were selected. It did not produce a consistent incremental AUC or KS gain. This is a mixed result: representation filtering contributed reproducibility, especially for CatBoost, without guaranteeing stronger discrimination.

## 6. Forward transfer: Home Credit-trained corrected CLIP → LendingClub v2

The forward matrix is `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet`. It contains 576 unique LendingClub v2 feature names, 32 saved joint dimensions, complete semantic-group labels, the corrected policy identifier, and the selected corrected checkpoint hash. The training manifests state that LendingClub v2 was external validation and was not used for training or checkpoint selection. The UMAP was newly calculated from this immutable matrix; no embedding was regenerated.

![Home Credit-trained corrected CLIP projected to LendingClub v2](figures/figure_3_forward_homecredit_to_lendingclub_umap.png)

*Figure 3. Home Credit-trained corrected CLIP projected to 576 LendingClub v2 features, colored by 17 LendingClub semantic groups. Coordinates were newly calculated from the saved 32-dimensional embedding matrix using cosine UMAP, 15 neighbours, minimum distance 0.1, two components, and seed 42. UMAP axes have no direct substantive interpretation. Source: `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet`.*

Original-space cosine kNN purity at k=10 is {forward.cosine_knn_purity:.6f}. With labels shuffled {PERMUTATIONS} times and embeddings fixed, the mean is {forward.shuffled_mean:.6f}, the 95th percentile is {forward.shuffled_p95:.6f}, and the 97.5th percentile is {forward.shuffled_p975:.6f}. UMAP trustworthiness at k=10 is {forward.umap_trustworthiness:.6f}. The observed local agreement is far above chance and the two-dimensional plot preserves local neighbourhoods well.

This supports forward representation transfer: a Home Credit-trained mapping organizes LendingClub features in a way that corresponds to LendingClub semantic labels. It does not support a downstream performance claim because no valid corrected Home Credit → LendingClub downstream prediction run is available. Forward representation success is therefore stronger than forward predictive evidence.

## 7. Reverse transfer: LendingClub v2-trained corrected CLIP → Home Credit

The reverse matrix is `results/corrected_lendingclub_to_homecredit_transfer/reverse_projection/homecredit_reverse_embeddings.parquet`. All five seed files exist and contain the same 436 feature IDs. Their spaces were aligned to seed 11 using orthogonal Procrustes alignment and aggregated by normalized mean. Feature IDs reconcile one-to-one to the saved Home Credit semantic metadata. Of 531 reverse-reconciliation records, 436 are compatible and 95 are excluded for missing semantic text embeddings.

![LendingClub v2-trained corrected CLIP projected to Home Credit](figures/figure_4_reverse_lendingclub_to_homecredit_umap.png)

*Figure 4. Five-seed LendingClub v2-trained corrected CLIP consensus projected without refitting to 436 Home Credit features, colored by 11 represented Home Credit semantic groups. UMAP settings match Figure 3. Axes have no direct substantive interpretation, and coordinates cannot be compared geometrically with Figure 3. Source: `results/corrected_lendingclub_to_homecredit_transfer/reverse_projection/homecredit_reverse_embeddings.parquet`.*

Reverse original-space kNN purity is {reverse.cosine_knn_purity:.6f}; the shuffled mean is {reverse.shuffled_mean:.6f}, the 95th percentile is {reverse.shuffled_p95:.6f}, and the 97.5th percentile is {reverse.shuffled_p975:.6f}. Trustworthiness is {reverse.umap_trustworthiness:.6f}. This supports reverse representation transfer. The higher numerical purity than the forward direction must not be treated as a ranking because group definitions, counts, and class imbalance differ materially.

Downstream results are weaker. LR reverse transfer produced OOT AUC {fmt(lr_rev.oot_auc)}, KS {fmt(lr_rev.oot_ks)}, PSI {fmt(lr_rev.score_psi)}, and Jaccard {fmt(lr_rev.mean_pairwise_jaccard)}. CatBoost produced AUC {fmt(cb_rev.oot_auc)}, KS {fmt(cb_rev.oot_ks)}, PSI {fmt(cb_rev.score_psi)}, and Jaccard {fmt(cb_rev.mean_pairwise_jaccard)}. The direct saved-prediction PSI values supersede older `model_score_psi.csv` scalars that used a different historical calculation.

![Reverse-transfer performance and stability](figures/figure_5_reverse_transfer_performance_stability.png)

*Figure 5. Reverse-transfer OOT AUC, pooled-OOF-reference score PSI, and corrected pairwise Jaccard. Sources: direct prediction metrics and corrected stability files under `results/corrected_lendingclub_to_homecredit_transfer/downstream/`.*

Reverse LR is not useful as a competitive predictor: its AUC is only {fmt(lr_rev.oot_auc)}. CatBoost recovers more discrimination but remains {cb_base.oot_auc - cb_rev.oot_auc:.6f} below the Home Credit mRMR baseline and {cb_comb.oot_auc - cb_rev.oot_auc:.6f} below the combined Task 1 pipeline. Reverse transfer is therefore technically valid and representationally structured, but not competitive downstream.

## 8. Bidirectional evidence synthesis

At the representation level, both directions pass a meaningful test. Each saved external matrix maps uniquely to feature identities and labels. In both spaces, observed kNN semantic purity is several times the shuffled-label upper reference, and UMAP trustworthiness is high. This is stronger evidence than visual clustering alone because the diagnostic is calculated in the original 32-dimensional space.

At the predictive level, the evidence is asymmetric. Reverse LendingClub → Home Credit has complete downstream OOF/OOT evaluation, but performance is weaker than within-Home-Credit alternatives. Forward Home Credit → LendingClub has an authenticated external embedding matrix but no corrected downstream prediction run. The absence of a forward model is not filled with scalar similarity scores or inferred from UMAP.

Accordingly, bidirectional representation evidence is stronger than bidirectional predictive evidence. General credit-feature signatures may have been learned at a moderate level, but robust bidirectional downstream prediction is not established. The most defensible wording is that corrected CLIP transferred semantic-statistical structure in both directions while competitive predictive transfer was not demonstrated.

## 9. Discussion

The main incremental experiment shows why layered methods must be evaluated rather than presumed superior. Adding a representation filter after the LLM made fold-selected sets much more reproducible, particularly for CatBoost. This suggests that corrected CLIP imposed a coherent ordering within the semantic candidate pool. Yet the predictive changes were negligible and inconsistent: LR AUC declined, CatBoost AUC rose by less than 0.001, and KS did not improve consistently.

One explanation is that the LLM pool already contained the strongest Home Credit signals, leaving little room for a representation layer to improve discrimination. Another is that CLIP optimizes semantic-statistical alignment, not target ranking. A feature can occupy a meaningful representation neighbourhood while contributing little additional outcome information after stronger variables enter the model. mRMR can also recover target relevance from a broader pool without requiring representation proximity.

The model difference is informative but not causal proof. CatBoost can exploit nonlinear thresholds and interactions, and it showed clearer stability gains than LR. Nevertheless, its tiny AUC change and lower KS prevent a claim that corrected CLIP improved prediction. The proper conclusion is model-dependent stability benefit, not model-dependent predictive superiority.

The directional diagnostics suggest that credit-feature concepts cross datasets despite different schemas. LendingClub groups such as revolving utilization, delinquency, account depth, inquiries, and FICO form a taxonomy different from Home Credit’s application, bureau, installment, previous-application, and credit-card aggregates. Because purity depends on taxonomy granularity and class balance, separate significance against shuffled labels is appropriate; direct ranking is not.

The reverse predictive weakness also demonstrates that representation coherence is not sufficient. Feature-level semantic alignment does not align borrower populations, underwriting regimes, target definitions, missingness processes, or temporal drift. A transferred ranking may identify intelligible features while failing to preserve the dataset-specific relationship between those features and default.

### Evidence reconciliation and practical interpretation

The audit also illustrates why provenance can change the interpretation of an apparently simple metric table. The central reusable registry correctly identifies the required runs and reproduces their OOT values, but it is still an index rather than the most direct evidence. For each required row, this revision opened the saved OOT prediction file, counted its records, and recalculated AUC and KS. It then matched those values to the run summary and registry. This process protects the comparison from similarly named invalid historical CLIP runs and from accidental substitution of aggregate fold values for pooled predictions.

The reverse-transfer PSI discrepancy is a concrete example. The legacy scalar `results/model_score_psi.csv` reflects the older final-fit-score calculation, whereas the later authenticated prediction-metric manifest derives PSI from saved pooled DEV OOF probabilities and saved OOT probabilities. The direct saved-prediction definition has stronger provenance and is used here. Both values remain scientifically explainable within their scopes, but they answer different drift questions. Reporting one without its reference population would create a false contradiction.

The stability reconciliation is equally important. Universe-dependent stability measures can look precise while using the wrong denominator. The Task 1 fold selections are real, and their overlaps are reproducible, but the stored Nogueira/Kuncheva calculation retained the 529-feature upstream universe after screening reduced the selectable set to 60 or 100. Pairwise Jaccard avoids that denominator and exactly reproduces from the five saved fold sets. Consequently, the conclusion that corrected CLIP improved selection reproducibility is supported even though the invalid universe-dependent measures are withheld.

For practical model development, the result argues against automatically adding representation layers to a strong semantic-plus-supervised pipeline. If the objective is discrimination, the combined method needs a prospectively defined minimum gain and paired prediction provenance. If the objective includes governance, reproducible feature sets, or resistance to fold-specific choices, the Jaccard improvement may still be operationally valuable. Those objectives should be declared before model selection rather than retrofitted after observing small AUC changes.

The transfer results suggest a similarly staged decision rule. Representation diagnostics can establish that projection is coherent enough to inspect, but downstream deployment requires a separate threshold for discrimination and temporal behaviour on the external dataset. Here the representation gate is passed in both directions; the predictive gate is not. This distinction prevents a visually persuasive embedding from being promoted into an unsupported model-performance claim.

## 10. Limitations

Task 1 lacks stable borrower IDs and pooled DEV OOF predictions. OOT metrics are exact, but paired method tests and authenticated Task 1 AUC drops are unavailable. Tiny AUC differences cannot be labelled significant or equivalent. The historical PSI uses final DEV-fit scores as reference rather than pooled OOF scores, so Task 1 PSI is comparable within Task 1 pairs but not directly to reverse-transfer PSI.

Task 1’s saved Nogueira and Kuncheva values used the wrong universe size and were excluded. Jaccard remains valid because it depends only on saved fold sets. Corrected CLIP candidate manifests show 60 and 100 actual candidates even where internal stability summaries retain 529.

UMAP is stochastic dimensionality reduction. A fixed seed and common settings make each plot reproducible, but absolute coordinates, rotation, cluster position, and cross-panel location have no substantive interpretation. Quantitative claims rely on original-space kNN purity and shuffled-label controls. Different taxonomies prevent a simple forward-versus-reverse quality ranking.

Forward corrected downstream prediction is absent. Reverse prediction alone cannot establish bidirectional predictive robustness. The package does not reconstruct missing predictions, generate new embeddings, or treat row order as identity. It also does not estimate causal contributions of the LLM, corrected CLIP, or mRMR beyond the saved pipeline contrasts.

## 11. Conclusion

LLM → mRMR is a strong compact Home Credit pipeline, but the full 529-feature mRMR baseline remains descriptively stronger in OOT AUC. Adding corrected CLIP after LLM did not provide a clear incremental predictive gain for either model. LR showed substantially higher feature-selection Jaccard but slightly worse score PSI; CatBoost showed better score and feature-selection stability with only a tiny, untested AUC increase and a small KS decrease. Corrected CLIP therefore contributed representation structure and selector reproducibility without guaranteeing stronger discrimination.

Both transfer directions contain non-random semantic structure in their saved embedding spaces. Reverse downstream prediction is not competitive, and forward downstream evidence is unavailable. The study supports bidirectional representation transfer, not bidirectional predictive robustness.

## Appendix A. Complete result table

The machine-readable table is [final_results_tables.csv](final_results_tables.csv). Empty DEV OOF fields are intentional and prevent mean fold AUC from being substituted for pooled OOF AUC.

## Appendix B. Selected features

Final selected-feature artifacts remain immutable at the source paths recorded in `artifact_inventory.md`. LR uses 20 features and CatBoost 40 for every required pipeline. The report package does not copy or modify those scientific files. Common high-ranked Home Credit signals include external scores, installment repayment behaviour, bureau debt and history, application amounts, and employment or age variables.

## Appendix C. Stable-core members and top-ranked features

The Home Credit stable-core source is `results/corrected_homecredit_clip/stable_core/anchor_members.csv`. The LendingClub reverse anchor source is `results/corrected_lendingclub_to_homecredit_transfer/source_anchor/anchor_members.csv`. Forward learned rankings remain in `results/corrected_homecredit_clip/training/lendingclub_v2_learned_scores.csv`; reverse rankings remain in `results/corrected_lendingclub_to_homecredit_transfer/reverse_projection/homecredit_reverse_scores.csv`. These rankings are representation outputs, not substitutes for downstream performance.

## Appendix D. Directional UMAP settings

Both plots use umap-learn {UMAP_SETTINGS["package_version"]}, cosine distance, 15 neighbours, minimum distance 0.1, two components, and random seed 42. Coordinates were newly calculated from saved embeddings. kNN purity uses the validated Task 3 operational implementation at k=10. Shuffled references use {PERMUTATIONS} label permutations with seed {PERMUTATION_SEED}. Exact coordinates and manifests are under `diagnostics/`.

## Appendix E. Claims matrix

The complete rating, evidence, contrary evidence, limitation, allowed wording, and prohibited wording matrix is [claims_and_evidence.csv](claims_and_evidence.csv).
"""


def claims_table(
    table: pd.DataFrame, directional: pd.DataFrame
) -> pd.DataFrame:
    lookup = table.set_index(["pipeline", "model"])
    fwd, rev = directional.iloc[0], directional.iloc[1]
    claims = [
        (
            "A",
            "LLM semantic screening produced a useful candidate pool.",
            "moderate",
            "LLM → mRMR OOT AUC was 0.738098 (LR) and 0.762954 (CatBoost).",
            "Both rows were below the full mRMR baseline.",
            "No isolated causal test of LLM screening; no pooled OOF.",
            "The LLM screen produced a useful compact candidate pool.",
            "The LLM screen was superior to full-pool selection.",
        ),
        (
            "B",
            "Corrected CLIP learned meaningful semantic-statistical structure.",
            "strong",
            (
                f"Forward/reverse purity {fwd.cosine_knn_purity:.6f}/"
                f"{rev.cosine_knn_purity:.6f}, both above shuffled p97.5."
            ),
            "Embedding structure did not ensure competitive downstream prediction.",
            "Purity taxonomies differ; UMAP is not predictive evidence.",
            "Corrected CLIP learned non-random semantic-statistical structure.",
            "UMAP proves predictive utility.",
        ),
        (
            "C",
            "LLM → corrected CLIP → mRMR outperformed LLM → mRMR for LR.",
            "not supported",
            "KS increased by only 0.000533.",
            "AUC decreased by 0.001102; no paired test.",
            "Stable borrower IDs are absent.",
            "LR predictive results were mixed with no clear gain.",
            "The combined LR pipeline outperformed LLM → mRMR.",
        ),
        (
            "D",
            "LLM → corrected CLIP → mRMR outperformed LLM → mRMR for CatBoost.",
            "weak",
            "AUC increased by 0.000866.",
            "KS decreased by 0.000878; the AUC change is tiny and untested.",
            "Stable borrower IDs are absent.",
            "CatBoost showed a tiny descriptive AUC increase, not clear superiority.",
            "The combined CatBoost pipeline was superior or equivalent.",
        ),
        (
            "E",
            "Corrected CLIP improved score stability after LLM.",
            "moderate",
            "CatBoost PSI fell from 0.008286 to 0.004094.",
            "LR PSI rose from 0.003798 to 0.004898.",
            "Conclusion differs by model.",
            "Score stability improved for CatBoost but not LR.",
            "Corrected CLIP consistently improved score stability.",
        ),
        (
            "F",
            "Home Credit-trained corrected CLIP produced structured LendingClub embeddings.",
            "strong",
            (
                f"Purity {fwd.cosine_knn_purity:.6f} vs shuffled p97.5 "
                f"{fwd.shuffled_p975:.6f}; trustworthiness "
                f"{fwd.umap_trustworthiness:.6f}."
            ),
            "No corrected forward downstream prediction run.",
            "External-validation labels assess representation only.",
            "Forward LendingClub embeddings show non-random local structure.",
            "Forward UMAP proves predictive transfer.",
        ),
        (
            "G",
            "LendingClub-trained corrected CLIP produced structured Home Credit embeddings.",
            "strong",
            (
                f"Purity {rev.cosine_knn_purity:.6f} vs shuffled p97.5 "
                f"{rev.shuffled_p975:.6f}; trustworthiness "
                f"{rev.umap_trustworthiness:.6f}."
            ),
            "Reverse downstream performance was weak relative to Home Credit pipelines.",
            "Ninety-five reconciliation records were ineligible.",
            "Reverse Home Credit embeddings show non-random local structure.",
            "Reverse UMAP proves model quality.",
        ),
        (
            "H",
            "Forward representation transfer was successful.",
            "strong",
            "Authenticated 576-row external matrix and strong original-space diagnostic.",
            "No forward downstream model.",
            "Success is representation-level only.",
            "Forward representation transfer was successful.",
            "Forward predictive transfer was successful.",
        ),
        (
            "I",
            "Reverse representation transfer was successful.",
            "strong",
            "Five seeds, frozen projection, 436 reconciled features, strong purity.",
            "Predictive competitiveness was not achieved.",
            "Success is representation-level.",
            "Reverse representation transfer was successful.",
            "Reverse representation success proves robust prediction.",
        ),
        (
            "J",
            "Reverse downstream LR transfer was useful.",
            "not supported",
            "The saved evaluation is technically valid.",
            "OOT AUC 0.573203 and KS 0.105769 are weak.",
            "Usefulness threshold was not prespecified.",
            "Reverse LR retained limited discrimination.",
            "Reverse LR transfer was useful or competitive.",
        ),
        (
            "K",
            "Reverse downstream CatBoost transfer was useful.",
            "weak",
            "OOT AUC 0.676603 and low pooled-OOF-reference PSI 0.017395.",
            "It remained far below Home Credit-trained alternatives.",
            "Usefulness threshold was not prespecified.",
            "CatBoost recovered moderate signal from the transferred pool.",
            "Reverse CatBoost was competitive.",
        ),
        (
            "L",
            "Reverse transfer was competitive.",
            "not supported",
            "CatBoost exceeded reverse LR.",
            "Both models trailed Home Credit-trained pipelines materially.",
            "No formal paired test, but gaps are large descriptively.",
            "Reverse transfer was weaker than within-dataset pipelines.",
            "Cross-dataset transfer was competitive.",
        ),
        (
            "M",
            "General credit-feature signatures were learned.",
            "moderate",
            "Both directional spaces exceed their shuffled-label references.",
            "Taxonomies differ and downstream transfer is weak.",
            "Feature-level evidence does not cover borrower-level relationships.",
            "The representation captured transferable credit-feature structure.",
            "A universal credit-risk representation was learned.",
        ),
        (
            "N",
            "Bidirectional downstream predictive robustness was established.",
            "not supported",
            "Representation evidence exists in both directions.",
            "Forward downstream evidence is absent; reverse prediction is not competitive.",
            "Only one direction has corrected downstream evaluation.",
            "Bidirectional representation evidence is stronger than predictive evidence.",
            "Bidirectional downstream predictive robustness was established.",
        ),
    ]
    return pd.DataFrame(
        claims,
        columns=[
            "claim_id",
            "claim",
            "rating",
            "supporting_evidence",
            "contrary_evidence",
            "limitations",
            "allowed_wording",
            "prohibited_wording",
        ],
    )


def professor_summary(
    table: pd.DataFrame, incremental: pd.DataFrame, directional: pd.DataFrame
) -> str:
    lookup = table.set_index(["pipeline", "model"])
    fwd, rev = directional.iloc[0], directional.iloc[1]
    return f"""
# Professor summary

This revision completes the requested comparison and directional representation analysis using only authenticated saved artifacts. It adds the missing standalone LLM → mRMR comparator, retains corrected CLIP → mRMR and LLM → corrected CLIP → mRMR, and separates Home Credit → LendingClub v2 from LendingClub v2 → Home Credit. No training, feature selection, prediction, or embedding generation was rerun.

## 1. LLM + corrected CLIP + mRMR

The direct question is: did adding corrected CLIP after LLM help?

For Logistic Regression, the answer is **not predictively**. LLM → mRMR had OOT AUC {lookup.loc[("LLM → mRMR", "LR"), "oot_auc"]:.6f} and KS {lookup.loc[("LLM → mRMR", "LR"), "oot_ks"]:.6f}. The combined pipeline had AUC {lookup.loc[("LLM → corrected CLIP → mRMR", "LR"), "oot_auc"]:.6f} and KS {lookup.loc[("LLM → corrected CLIP → mRMR", "LR"), "oot_ks"]:.6f}. AUC fell by {abs(incremental.iloc[0]["delta_oot_auc_combined_minus_llm"]):.6f}; KS rose by only {incremental.iloc[0]["delta_oot_ks_combined_minus_llm"]:.6f}. Score PSI increased from {lookup.loc[("LLM → mRMR", "LR"), "score_psi"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "LR"), "score_psi"]:.6f}, so score stability was slightly worse. Feature-selection Jaccard improved substantially, from {lookup.loc[("LLM → mRMR", "LR"), "mean_pairwise_jaccard"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "LR"), "mean_pairwise_jaccard"]:.6f}. LR therefore gained selection reproducibility but not prediction or score stability.

For CatBoost, the answer is **a stability benefit without clear predictive benefit**. AUC moved from {lookup.loc[("LLM → mRMR", "CatBoost"), "oot_auc"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "CatBoost"), "oot_auc"]:.6f}, an increase of only {incremental.iloc[1]["delta_oot_auc_combined_minus_llm"]:.6f}. KS decreased from {lookup.loc[("LLM → mRMR", "CatBoost"), "oot_ks"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "CatBoost"), "oot_ks"]:.6f}. The tiny, untested AUC change does not establish meaningful improvement. PSI fell from {lookup.loc[("LLM → mRMR", "CatBoost"), "score_psi"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "CatBoost"), "score_psi"]:.6f}, and Jaccard rose from {lookup.loc[("LLM → mRMR", "CatBoost"), "mean_pairwise_jaccard"]:.6f} to {lookup.loc[("LLM → corrected CLIP → mRMR", "CatBoost"), "mean_pairwise_jaccard"]:.6f}.

The result is mixed rather than uniformly positive. Corrected CLIP made selected feature sets more reproducible, especially for CatBoost, but did not consistently improve discrimination. The full Home Credit mRMR baseline remained descriptively higher in OOT AUC for both models.

The stability comparison required a provenance correction. The saved Task 1 Nogueira and Kuncheva values identify 529 as the selectable universe, although the actual LLM and combined candidate manifests contain 60 LR and 100 CatBoost features. Those universe-dependent values are not used. Mean pairwise Jaccard was recomputed directly from the five saved fold feature sets and matched the stored Jaccard values. Task 1 PSI is directly comparable within each model pair because both methods use the same DEV/OOT windows and historical calculation, but it is not compared numerically with reverse-transfer PSI, whose reference is pooled DEV OOF predictions.

## 2. Reverse transfer

The LendingClub v2-trained corrected CLIP representation was projected frozen to Home Credit using five aligned seed spaces. It produced 436 eligible Home Credit embeddings and fixed 60/100 candidate pools before DEV-only mRMR. Reverse LR had OOT AUC {lookup.loc[("LendingClub corrected CLIP → Home Credit", "LR"), "oot_auc"]:.6f}; reverse CatBoost had {lookup.loc[("LendingClub corrected CLIP → Home Credit", "CatBoost"), "oot_auc"]:.6f}. CatBoost recovered more signal and had low PSI, but both models were weaker than Home Credit-trained pipelines. Reverse predictive transfer was therefore validly evaluated but not competitive.

The reverse evaluation has the strongest row provenance in the study: 82,647 pooled DEV OOF predictions and 120,053 OOT predictions retain stable `SK_ID_CURR` values and reconcile to prediction and metric manifests. LR OOT performance was weak, while CatBoost retained moderate signal but remained roughly 0.090 AUC below the Home Credit mRMR baseline. Low score PSI does not overcome that discrimination gap.

## 3. Directional embedding evidence

The forward UMAP represents 576 LendingClub v2 features projected through the Home Credit-trained corrected representation. Original-space kNN purity was {fwd.cosine_knn_purity:.6f}, versus shuffled mean {fwd.shuffled_mean:.6f} and 97.5th percentile {fwd.shuffled_p975:.6f}; trustworthiness was {fwd.umap_trustworthiness:.6f}.

The reverse UMAP represents 436 Home Credit features projected through the five-seed LendingClub-trained consensus. Purity was {rev.cosine_knn_purity:.6f}, versus shuffled mean {rev.shuffled_mean:.6f} and 97.5th percentile {rev.shuffled_p975:.6f}; trustworthiness was {rev.umap_trustworthiness:.6f}. Both directions show non-random representation structure. Their taxonomies differ, so purity values and UMAP locations must not be ranked directly.

The scientific distinction is decisive: representation success is supported in both directions; predictive success is not. Stability improved after corrected CLIP mainly for CatBoost. Bidirectional robustness remains unsupported because the forward direction has no corrected downstream model and reverse performance is not competitive.

All directional coordinates were newly calculated from the already-saved 32-dimensional matrices using the same cosine UMAP settings: 15 neighbours, minimum distance 0.1, two components, and seed 42. The original embeddings were not regenerated. UMAP axes and cluster locations have no direct substantive interpretation, and the two layouts are not geometrically comparable. The original-space kNN diagnostics, rather than visual position, support the representation conclusion.
"""


def cover_message(table: pd.DataFrame, incremental: pd.DataFrame) -> str:
    return f"""
# Professor cover message

Professor,

All requested reporting and diagnostic analyses are complete in the revised v2 package. The Home Credit comparison now includes the missing standalone LLM → mRMR pipeline alongside corrected CLIP → mRMR, LLM → corrected CLIP → mRMR, the mRMR baseline, and reverse transfer for both Logistic Regression and CatBoost.

Adding corrected CLIP after LLM produced a mixed result. Logistic Regression did not gain predictive performance: OOT AUC decreased by {abs(incremental.iloc[0]["delta_oot_auc_combined_minus_llm"]):.6f}, although fold-to-fold feature overlap improved. CatBoost AUC increased by only {incremental.iloc[1]["delta_oot_auc_combined_minus_llm"]:.6f} while KS decreased; this is not a clear predictive gain. CatBoost did show better score and feature-selection stability.

The package also includes separate UMAPs and original-space diagnostics for Home Credit-trained corrected CLIP projected to LendingClub v2 and LendingClub v2-trained corrected CLIP projected to Home Credit. Both directions show non-random representation structure. Reverse downstream transfer remained weaker than Home Credit-trained pipelines, and the forward direction has no valid corrected downstream model. Bidirectional representation evidence is therefore stronger than bidirectional predictive evidence; bidirectional predictive robustness remains unsupported.
"""


def source_artifacts(table: pd.DataFrame) -> list[dict]:
    paths = set(table["_prediction_path"]) | set(table["source_metric_path"])
    paths |= set(table["_stability_path"])
    paths |= {
        "results/research_summary/reusable_metrics.csv",
        "results/research_summary/run_index.csv",
        "results/research_summary/artifact_registry.csv",
        "results/research_summary/selected_feature_registry.csv",
        (
            "results/corrected_homecredit_clip/combined_pipeline/"
            "combined_pipeline_manifest.json"
        ),
        (
            "results/corrected_homecredit_clip/training/"
            "lendingclub_v2_joint_embeddings.parquet"
        ),
        "results/corrected_homecredit_clip/training/training_manifest.json",
        (
            "results/corrected_lendingclub_to_homecredit_transfer/"
            "reverse_projection/homecredit_reverse_embeddings.parquet"
        ),
        (
            "results/corrected_lendingclub_to_homecredit_transfer/"
            "reverse_projection/reverse_projection_manifest.json"
        ),
        (
            "results/corrected_homecredit_clip/feature_universe/"
            "feature_universe_reconciliation.csv"
        ),
    }
    records = []
    for path in sorted(paths):
        absolute = ROOT / path
        assert absolute.exists(), path
        records.append(
            {
                "path": path,
                "sha256": sha256(absolute),
                "size_bytes": absolute.stat().st_size,
            }
        )
    return records


def reproducibility_text(
    directional: pd.DataFrame, sources: list[dict]
) -> str:
    fwd, rev = directional.iloc[0], directional.iloc[1]
    return f"""
# Reproducibility summary

## Scope

This v2 package is a report-only derivative. It reads immutable saved metrics, predictions, fold selections, manifests, and embedding matrices. It does not execute prepare, train, project, evaluate, register, model fitting, mRMR, prediction generation, or embedding generation.

## Metric authentication

For all ten required pipeline/model rows, OOT AUC and KS were independently recomputed from saved row-level predictions and matched the run summary and central reusable-metrics registry within `1e-12`. All OOT files contain 120,053 rows. Task 1 files have no stable borrower IDs and no saved pooled DEV OOF predictions; those fields remain empty. Reverse-transfer DEV OOF and OOT metrics use authenticated stable IDs and prediction manifests.

Task 1 PSI uses final DEV-fit in-sample scores as reference and OOT scores as comparison through the shared `calculate_psi` implementation. It is comparable within the direct LLM-only versus combined pairs. Reverse PSI uses pooled DEV OOF probabilities, DEV-OOF quantile bins, and OOT comparison probabilities. These scopes are not compared numerically across tasks.

Pairwise Jaccard was recomputed from the five saved fold-selected feature sets and matched each saved scalar. Task 1 Nogueira/Kuncheva values were excluded because their files used universe size 529 rather than the authenticated candidate pools of 60 for LR and 100 for CatBoost.

## Forward directional visualization

- Source representation: corrected Home Credit-trained CLIP, selected corrected seed-55 checkpoint.
- Projected dataset: LendingClub v2.
- Embedding source: `{fwd.source_embedding_path}`.
- Valid features: {fwd.valid_embedding_count}; semantic groups: {fwd.semantic_group_count}.
- Semantic labels: embedded row metadata authenticated against the external-pair count.
- Coordinates: newly calculated from saved embeddings; no embedding generation.

## Reverse directional visualization

- Source representation: five-seed corrected LendingClub v2-trained CLIP consensus.
- Projected dataset: Home Credit.
- Embedding source: `{rev.source_embedding_path}`.
- Valid features: {rev.valid_embedding_count}; excluded reconciliation records: {rev.excluded_feature_count}.
- Alignment: orthogonal Procrustes to seed 11; normalized mean consensus.
- Semantic labels: `results/corrected_homecredit_clip/feature_universe/feature_universe_reconciliation.csv`, joined one-to-one by feature ID.
- Coordinates: newly calculated from saved embeddings.

## Common diagnostic procedure

Both directions use umap-learn {UMAP_SETTINGS["package_version"]}, cosine metric, 15 neighbours, minimum distance 0.1, two components, and random seed 42. Original-space kNN purity uses the validated Task 3 operational procedure at k=10. Labels are shuffled {PERMUTATIONS} times with NumPy seed {PERMUTATION_SEED} while embeddings and neighbour identities remain fixed. Trustworthiness uses k=10 and cosine distance.

The source hashes are recorded in `final_package_manifest.json` and summarized in `artifact_inventory.md`. Output hashes cover every generated file except the package manifest itself. The manifest intentionally has no self-hash.

## Reproduction command

From the repository root:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\build_final_research_package_v2.py
```

The command rebuilds only `results/final_research_package_v2/`. It does not alter the v1 package or any scientific source artifact.
"""


def inventory_text(sources: list[dict]) -> str:
    rows = "\n".join(
        f"| `{item['path']}` | `{item['sha256']}` | {item['size_bytes']:,} |"
        for item in sources
    )
    return f"""
# Artifact inventory

All entries below are read-only scientific or registry sources used by the v2 reporting process. Generated output hashes are in `final_package_manifest.json`.

| Source artifact | SHA-256 | Bytes |
|---|---|---:|
{rows}

## Directional identity

The forward UMAP uses LendingClub v2 feature rows from `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet`, produced by the corrected Home Credit-trained representation. It is not the within-domain Home Credit UMAP.

The reverse UMAP uses Home Credit feature rows from `results/corrected_lendingclub_to_homecredit_transfer/reverse_projection/homecredit_reverse_embeddings.parquet`, produced by the frozen five-seed LendingClub v2-trained consensus. The five seed matrices and alignment details are authenticated in the reverse diagnostic manifest.

Newly generated coordinates, plots, tables, and prose are report diagnostics derived from immutable sources. They are not scientific stage artifacts.
"""


def final_results_csv(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pipeline",
        "source_dataset",
        "evaluation_dataset",
        "model",
        "candidate_pool_size",
        "final_feature_count",
        "dev_oof_rows",
        "dev_oof_auc",
        "oot_rows",
        "oot_auc",
        "oot_ks",
        "auc_drop",
        "score_psi",
        "psi_scope",
        "nogueira_stability",
        "mean_pairwise_jaccard",
        "prediction_provenance",
        "evidence_status",
        "source_metric_path",
        "notes",
    ]
    return table[columns]


def json_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(
        orient="records"
    )


def validate_outputs(expected_files: list[Path]) -> dict:
    for path in expected_files:
        assert path.exists() and path.stat().st_size > 0, path
    for path in OUT.rglob("*.csv"):
        pd.read_csv(path)
    for path in OUT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "TODO" not in text and "PLACEHOLDER" not in text
        for target in re.findall(r"!?\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target).resolve().exists(), (path, target)
    report_words = len(
        (OUT / "final_research_report.md").read_text(encoding="utf-8").split()
    )
    professor_words = len(
        (OUT / "professor_summary.md").read_text(encoding="utf-8").split()
    )
    cover_words = len(
        (OUT / "professor_cover_message.md").read_text(encoding="utf-8").split()
    )
    assert 3500 <= report_words <= 5000, report_words
    assert 600 <= professor_words <= 900, professor_words
    assert 140 <= cover_words <= 220, cover_words
    return {
        "all_required_files_exist": True,
        "all_csv_files_parse": True,
        "markdown_links_resolve": True,
        "placeholder_text_absent": True,
        "report_word_count": report_words,
        "professor_summary_word_count": professor_words,
        "professor_cover_message_word_count": cover_words,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)

    table = authenticate_task1()
    incremental = incremental_table(table)
    directional = authenticate_embeddings()
    plot_task1(table, incremental)

    final_results_csv(table).to_csv(
        OUT / "final_results_tables.csv", index=False
    )
    incremental.to_csv(OUT / "task1_incremental_comparison.csv", index=False)
    directional.to_csv(
        OUT / "directional_embedding_diagnostics.csv", index=False
    )
    claims = claims_table(table, directional)
    assert set(claims["rating"]) <= {"strong", "moderate", "weak", "not supported"}
    claims.to_csv(OUT / "claims_and_evidence.csv", index=False)

    write_text(
        OUT / "final_research_report.md",
        report_markdown(table, incremental, directional),
    )
    write_text(
        OUT / "professor_summary.md",
        professor_summary(table, incremental, directional),
    )
    write_text(
        OUT / "professor_cover_message.md",
        cover_message(table, incremental),
    )
    sources = source_artifacts(table)
    write_text(
        OUT / "reproducibility_summary.md",
        reproducibility_text(directional, sources),
    )
    write_text(OUT / "artifact_inventory.md", inventory_text(sources))

    generated_without_manifest = sorted(
        [
            path
            for path in OUT.rglob("*")
            if path.is_file() and path.name != "final_package_manifest.json"
        ],
        key=lambda path: rel(path),
    )
    expected_names = {
        "final_research_report.md",
        "professor_summary.md",
        "professor_cover_message.md",
        "final_results_tables.csv",
        "task1_incremental_comparison.csv",
        "directional_embedding_diagnostics.csv",
        "claims_and_evidence.csv",
        "reproducibility_summary.md",
        "artifact_inventory.md",
        "figures/figure_1_homecredit_pipeline_oot_auc.png",
        "figures/figure_2_task1_incremental_value.png",
        "figures/figure_3_forward_homecredit_to_lendingclub_umap.png",
        "figures/figure_4_reverse_lendingclub_to_homecredit_umap.png",
        "figures/figure_5_reverse_transfer_performance_stability.png",
        "diagnostics/forward_homecredit_to_lendingclub_umap_coordinates.csv",
        "diagnostics/forward_homecredit_to_lendingclub_umap_manifest.json",
        "diagnostics/reverse_lendingclub_to_homecredit_umap_coordinates.csv",
        "diagnostics/reverse_lendingclub_to_homecredit_umap_manifest.json",
    }
    assert {
        path.relative_to(OUT).as_posix() for path in generated_without_manifest
    } == expected_names
    validation = validate_outputs(generated_without_manifest)

    manifest = {
        "package_version": "2.0",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_audit_status": "passed",
        "generated_files": [
            {
                "path": rel(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated_without_manifest
        ],
        "source_artifacts": sources,
        "authenticated_metric_summary": json_records(final_results_csv(table)),
        "task1_incremental_value_summary": json_records(incremental),
        "directional_embedding_diagnostic_summary": json_records(directional),
        "known_limitations": [
            "Task 1 lacks stable borrower IDs and pooled DEV OOF predictions.",
            (
                "Task 1 saved Nogueira/Kuncheva values used an invalid N=529 "
                "universe and are omitted; pairwise Jaccard is used."
            ),
            (
                "Task 1 PSI and reverse-transfer PSI have different reference "
                "scopes and are not directly compared."
            ),
            (
                "Forward corrected downstream prediction is unavailable; "
                "bidirectional predictive robustness is unsupported."
            ),
            (
                "Directional semantic taxonomies differ, and separate UMAP "
                "coordinates are not geometrically comparable."
            ),
        ],
        "validation": validation,
        "scientific_source_mutations": [],
        "manifest_self_hash": None,
        "manifest_self_hash_note": (
            "Intentionally omitted; no canonical self-hash method is defined."
        ),
        "pdf_readiness_status": "READY FOR PROFESSOR PDF EXPORT",
    }
    manifest_path = OUT / "final_package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_outputs(generated_without_manifest + [manifest_path])
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
