from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.checkpointing import load_checkpoint
from credit_risk_fs.clip.training_validation import (
    load_and_validate_training_inputs,
    load_training_config,
    tensors_for_pairs,
)
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


ROOT = Path("results/corrected_homecredit_clip")
SEEDS = [11, 22, 33, 44, 55]


def main() -> None:
    config = load_training_config("configs/corrected_homecredit_clip/training.yaml")
    data = load_and_validate_training_inputs(config)
    pairs = data.homecredit_pairs.copy()
    text, stat = tensors_for_pairs(pairs, data.homecredit_text, data.homecredit_stat)
    seed_embeddings: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        seed_dir = ROOT / "training" / "seeds" / f"seed_{seed}"
        model = load_checkpoint(
            checkpoint_path=seed_dir / "best_checkpoint.pt",
            manifest_path=seed_dir / "checkpoint_manifest.json",
            config=config,
            upstream_hashes=data.upstream_hashes,
            map_location="cpu",
        )
        with torch.no_grad():
            a, b = model(text, stat)
            joint = torch.nn.functional.normalize((a + b) / 2, p=2, dim=1)
        seed_embeddings[seed] = joint.numpy()

    reference = seed_embeddings[SEEDS[0]]
    aligned = []
    for seed in SEEDS:
        values = seed_embeddings[seed]
        if seed != SEEDS[0]:
            rotation, _ = orthogonal_procrustes(values, reference)
            values = values @ rotation
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        aligned.append(values)
    consensus = np.mean(aligned, axis=0)
    consensus /= np.maximum(np.linalg.norm(consensus, axis=1, keepdims=True), 1e-12)

    _feature_universe(pairs)
    embedding_dir = ROOT / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    meta = pairs[["feature_id", "feature_name", "split", "semantic_group", "source_table_or_formula"]].copy()
    emb = pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(consensus, columns=[f"embedding_{i:04d}" for i in range(32)])],
        axis=1,
    )
    emb.to_parquet(embedding_dir / "feature_embeddings.parquet", index=False)
    emb.to_csv(embedding_dir / "feature_embeddings.csv", index=False)
    (embedding_dir / "embedding_definition.md").write_text(
        "# Corrected feature embedding\n\n"
        "For each seed, the feature vector is the L2-normalized mean of matched projected semantic and "
        "statistical views. Seed 11 is the predeclared alignment reference; seeds 22/33/44/55 are aligned "
        "to it by deterministic orthogonal Procrustes, averaged, and L2-normalized. Dimensionality: 32.\n",
        encoding="utf-8",
    )
    _write_json(embedding_dir / "embedding_manifest.json", {
        "seeds": SEEDS, "fusion_rule": "normalized mean of projected views",
        "seed_aggregation": "orthogonal-Procrustes alignment to seed 11 then normalized mean",
        "pairing_policy_version": "identity_equivalence_v2", "feature_count": len(emb),
    })

    _diagnostics(emb, consensus, aligned)
    _anchor(emb, consensus, aligned)
    _training_tables()


def _feature_universe(pairs: pd.DataFrame) -> None:
    out = ROOT / "feature_universe"
    out.mkdir(parents=True, exist_ok=True)
    expected = [line.strip() for line in Path("results/_candidate_columns.txt").read_text().splitlines() if line.strip()]
    evidence = pd.read_csv("results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv")
    evidence = evidence.rename(columns={"feature": "feature_name"}).drop_duplicates("feature_name")
    text_names = set(pd.read_csv("results/clip/text_baseline/homecredit_feature_text.csv").feature_name.astype(str))
    stat = pd.read_parquet("results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet")
    stat_names = set(stat.feature_name.astype(str))
    pair_names = set(pairs.feature_name.astype(str))
    rows = []
    for idx, name in enumerate(expected):
        row = evidence[evidence.feature_name.astype(str).eq(name)]
        info = row.iloc[0] if len(row) else {}
        text_ok, stat_ok = name in text_names, name in stat_names
        rows.append({
            "feature_id": sha256_text(f"homecredit|{name}"),
            "feature_name": name,
            "source_table": info.get("source_table", "unavailable"),
            "semantic_group": info.get("semantic_group", "unavailable"),
            "present_in_expected_529": True,
            "text_view_available": text_ok,
            "statistical_view_available": stat_ok,
            "eligible_for_clip_training": name in pair_names and str(stat.loc[stat.feature_name.eq(name), "split"].iloc[0]) == "train" if name in stat_names else False,
            "eligible_for_projection": name in pair_names,
            "exclusion_reason": "" if name in pair_names else str(info.get("clip_training_exclusion_reason", "missing_complete_frozen_views")),
            "embedding_available": name in pair_names,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "feature_universe_reconciliation.csv", index=False)
    _write_json(out / "feature_universe_summary.json", {
        "expected_total": 529, "metadata_total": int(evidence.feature_name.nunique()),
        "text_view_total": len(text_names), "statistical_view_total": len(stat_names),
        "paired_training_total": int((frame.eligible_for_clip_training).sum()),
        "projected_embedding_total": int(frame.embedding_available.sum()),
        "umap_total": int(frame.embedding_available.sum()),
        "excluded_total_by_reason": frame.loc[~frame.embedding_available, "exclusion_reason"].value_counts().to_dict(),
    })


def _diagnostics(emb: pd.DataFrame, values: np.ndarray, aligned: list[np.ndarray]) -> None:
    out = ROOT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    labels = emb.semantic_group.astype(str).to_numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(values)
    coord = emb[["feature_id", "feature_name", "semantic_group"]].copy()
    coord["umap_1"], coord["umap_2"] = coords[:, 0], coords[:, 1]
    coord.to_csv(out / "umap_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    for group in sorted(set(labels)):
        mask = labels == group
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=.75, label=group)
    ax.set_title(f"Corrected Home Credit CLIP embeddings (n={len(values)})")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "homecredit_feature_umap.png", dpi=220)
    fig.savefig(out / "homecredit_feature_umap.pdf")
    plt.close(fig)

    nbr = NearestNeighbors(n_neighbors=11, metric="cosine").fit(values)
    inds = nbr.kneighbors(return_distance=False)[:, 1:]
    purity = np.mean(labels[inds] == labels[:, None], axis=1)
    pd.DataFrame({"feature_name": emb.feature_name, "semantic_group": labels, "knn_purity_k10": purity}).to_csv(out / "neighbour_purity.csv", index=False)
    sim = values @ values.T
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(values), dtype=bool)
    observed = {
        "cosine_knn_purity_k10": float(purity.mean()),
        "silhouette_cosine": float(silhouette_score(values, labels, metric="cosine")),
        "within_group_cosine": float(sim[same & off].mean()),
        "between_group_cosine": float(sim[(~same) & off].mean()),
        "umap_trustworthiness_k10": float(trustworthiness(values, coords, n_neighbors=10, metric="cosine")),
    }
    overlaps = []
    base = NearestNeighbors(n_neighbors=11, metric="cosine").fit(aligned[0]).kneighbors(return_distance=False)[:, 1:]
    for arr in aligned[1:]:
        other = NearestNeighbors(n_neighbors=11, metric="cosine").fit(arr).kneighbors(return_distance=False)[:, 1:]
        overlaps.append(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(base, other)]))
    observed["seed_neighbour_overlap_k10_mean"] = float(np.mean(overlaps))
    pd.DataFrame([{"metric": k, "value": v} for k, v in observed.items()]).to_csv(out / "cluster_metrics.csv", index=False)
    rng = np.random.default_rng(20260625)
    controls = []
    for i in range(200):
        shuffled = rng.permutation(labels)
        controls.append({"permutation": i, "knn_purity_k10": float(np.mean(shuffled[inds] == shuffled[:, None])),
                         "silhouette_cosine": float(silhouette_score(values, shuffled, metric="cosine"))})
    control = pd.DataFrame(controls)
    control.to_csv(out / "shuffled_label_control.csv", index=False)
    rating = "strong" if observed["cosine_knn_purity_k10"] > control.knn_purity_k10.quantile(.975) and observed["silhouette_cosine"] > control.silhouette_cosine.quantile(.975) else "moderate"
    (out / "diagnostics_summary.md").write_text(
        f"# Diagnostics summary\n\nObserved kNN purity: {observed['cosine_knn_purity_k10']:.4f}; "
        f"shuffle 97.5%: {control.knn_purity_k10.quantile(.975):.4f}. Observed silhouette: "
        f"{observed['silhouette_cosine']:.4f}; shuffle 97.5%: {control.silhouette_cosine.quantile(.975):.4f}. "
        f"Evidence rating: **{rating}**. This is representation structure, not predictive-value evidence.\n",
        encoding="utf-8",
    )


def _anchor(emb: pd.DataFrame, values: np.ndarray, aligned: list[np.ndarray]) -> None:
    out = ROOT / "stable_core"
    out.mkdir(parents=True, exist_ok=True)
    anchor_source = pd.read_csv("results/clip_v2/statistical_view/homecredit_statistical_anchor_features.csv")
    names = set(anchor_source.feature_name.astype(str))
    mask = emb.feature_name.astype(str).isin(names).to_numpy()
    anchor = values[mask].mean(axis=0); anchor /= np.linalg.norm(anchor)
    np.save(out / "stable_core_anchor_vector.npy", anchor)
    per_seed_ranks = []
    for arr in aligned:
        a = arr[mask].mean(axis=0); a /= np.linalg.norm(a)
        per_seed_ranks.append(pd.Series(-(arr @ a)).rank(method="first").to_numpy())
    ranks = np.vstack(per_seed_ranks)
    scores = values @ anchor
    order = np.argsort(-scores)[:20]
    top = emb.iloc[order][["feature_id", "feature_name", "source_table_or_formula", "semantic_group"]].copy()
    top = top.rename(columns={"source_table_or_formula": "source_table"})
    top.insert(0, "rank", range(1, 21))
    top["cosine_similarity"] = scores[order]
    top["seed_mean_rank"] = ranks[:, order].mean(axis=0)
    top["seed_rank_std"] = ranks[:, order].std(axis=0)
    top["top20_seed_frequency"] = (ranks[:, order] <= 20).mean(axis=0)
    top.to_csv(out / "top20_anchor_neighbours.csv", index=False)
    drift = pd.read_csv("results/homecredit/analysis/feature_level_drift/feature_level_psi_by_run.csv")
    drift = drift[~drift.selector.astype(str).str.contains("clip", case=False, na=False)]
    psi = drift.groupby("feature").psi_dev_oot.mean()
    evidence = pd.read_csv("results/homecredit/feature_level_evidence.csv").set_index("feature_name")
    rows = []
    for name in top.feature_name:
        rows.append({"feature_name": name, "feature_level_psi": psi.get(name, "unavailable"),
                     "missingness_shift": "unavailable", "distribution_shift_metric": psi.get(name, "unavailable"),
                     "selection_frequency_valid_baselines": evidence["mean_within_run_selection_frequency"].get(name, "unavailable") if name in evidence.index else "unavailable",
                     "rank_stability_valid_baselines": "unavailable",
                     "known_stable_flag": bool(name in names),
                     "evidence_source_path": "results/homecredit/analysis/feature_level_drift/feature_level_psi_by_run.csv;results/homecredit/feature_level_evidence.csv",
                     "evidence_status": "available" if name in psi.index else "unavailable"})
    pd.DataFrame(rows).to_csv(out / "anchor_neighbour_stability_evidence.csv", index=False)
    (out / "stable_core_anchor_definition.md").write_text(
        "# Stable-core anchor\n\nThe anchor is the L2-normalized centroid of the 23 frozen Home Credit "
        "training-split stable-core members in `results/clip_v2/statistical_view/homecredit_statistical_anchor_features.csv`. "
        "Membership is not recomputed from corrected neighbours. CLIP training uses no target or OOT data. "
        "Consensus aggregation follows the Procrustes-aligned five-seed rule.\n", encoding="utf-8")
    _write_json(out / "anchor_manifest.json", {"anchor_count": int(mask.sum()), "anchor_features": sorted(names),
        "pairing_policy_version": "identity_equivalence_v2", "target_used": False, "oot_used": False,
        "anchor_hash": sha256_text(anchor.tobytes().hex())})
    ranking = emb[["feature_name", "semantic_group", "source_table_or_formula"]].copy()
    ranking["consensus_clip_score"] = scores
    ranking = ranking.sort_values(["consensus_clip_score", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    ranking["consensus_clip_rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(ROOT / "combined_pipeline" / "corrected_consensus_clip_scores.csv", index=False)


def _training_tables() -> None:
    out = ROOT / "training"
    seeds = pd.read_csv(out / "seed_comparison.csv")
    seeds.to_csv(out / "seed_summary.csv", index=False)
    retrieval = pd.read_csv(out / "retrieval_metrics.csv")
    retrieval.to_csv(out / "alignment_metrics.csv", index=False)
    registry = seeds[["seed", "checkpoint_path", "checkpoint_hash", "best_epoch", "best_validation_loss", "best_validation_mrr"]].copy()
    registry["pairing_policy_version"] = "identity_equivalence_v2"
    registry["data_manifest_hash"] = sha256_file("results/clip/dry_run/training_manifest.json")
    registry.to_csv(out / "checkpoint_registry.csv", index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    (ROOT / "combined_pipeline").mkdir(parents=True, exist_ok=True)
    main()
