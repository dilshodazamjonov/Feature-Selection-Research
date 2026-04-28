import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import pandas as pd
from openai import OpenAI

from iv_woe_filter import IVWOEFilter
from feature_selection.missing_filter import MissingRateFilter
from utils.feature_metadata import build_feature_metadata
from utils.logging_config import setup_logging

dotenv.load_dotenv()

logger = setup_logging("llm_selector", level=logging.INFO)


class LLMSelector:
    """
    Fold-local LLM ranking selector with prompt-aware shared cache keys.

    The selector always builds ranking metadata from the current training slice
    only. It can generate a broad shared ranking and then truncate that ranking
    differently for LLM-only and hybrid use cases.
    """

    def __init__(
        self,
        description_csv_path: str,
        cache_dir: str = "results/_llm_rankings_cache",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_features: int = 100,
        ranking_budget: int | None = None,
        feature_budget: int | None = None,
        shared_ranking_enabled: bool = True,
        config_hash: str | None = None,
        prompt_version: str = "stability_expert_v3",
        ranking_budget_config: Dict[str, int] | None = None,
        shared_pool_size: int | None = None,
        lr_feature_budget: int = 20,
        catboost_feature_budget: int = 40,
        lr_candidate_pool_budget: int = 60,
        catboost_candidate_pool_budget: int = 100,
        max_missing_rate: float = 0.95,
        iv_filter_kwargs: Dict | None = None,
        feature_metadata: List[Dict] | None = None,
    ):
        self.description_csv_path = description_csv_path
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.temperature = temperature
        self.max_features = int(max_features)
        self.ranking_budget = int(ranking_budget or shared_pool_size or max_features)
        self.feature_budget = int(feature_budget or max_features)
        self.shared_ranking_enabled = shared_ranking_enabled
        self.config_hash = config_hash or "default"
        self.prompt_version = str(prompt_version)
        self.ranking_budget_config = dict(ranking_budget_config or {})
        self.shared_pool_size = int(shared_pool_size or self.ranking_budget)
        self.lr_feature_budget = int(lr_feature_budget)
        self.catboost_feature_budget = int(catboost_feature_budget)
        self.lr_candidate_pool_budget = int(lr_candidate_pool_budget)
        self.catboost_candidate_pool_budget = int(catboost_candidate_pool_budget)
        self.max_missing_rate = max_missing_rate
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})
        self.feature_metadata = feature_metadata

        self.ranked_features_: list[str] | None = None
        self.selected_features: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.artifact_dir: Path | None = None
        self.ranking_artifact_dir: Path | None = None
        self.scope: str = "global"
        self.fold_id: int | None = None
        self.metadata_signature_: str | None = None
        self.prompt_hash_: str | None = None
        self.cache_file_: Path | None = None
        self.cache_hit_: bool = False
        self.llm_calls_made_: int = 0
        self.llm_cache_hits_: int = 0
        self.selection_payload_: dict | None = None
        self._client: OpenAI | None = None
        self.missing_filter_: MissingRateFilter | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | os.PathLike) -> None:
        self.artifact_dir = Path(artifact_dir)

    def set_ranking_context(
        self,
        *,
        scope: str,
        fold_id: int | None = None,
        ranking_artifact_dir: str | os.PathLike | None = None,
        **_: object,
    ) -> None:
        self.scope = scope
        self.fold_id = fold_id
        if ranking_artifact_dir is not None:
            self.ranking_artifact_dir = Path(ranking_artifact_dir)

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API Key not found. Set OPENAI_API_KEY before running the LLM selector."
            )

        self._client = OpenAI(api_key=key)
        return self._client

    def _build_prompt(self, metadata: List[Dict]) -> str:
        lines = []
        for feature in metadata:
            name = feature["name"]
            desc = feature.get("description") or "No description available"
            table = feature.get("table") or "application_train"
            semantic_group = feature.get("semantic_group") or "other"
            missing = feature.get("missing_rate", 0.0)
            dtype = feature.get("dtype") or "unknown"
            non_null_count = feature.get("non_null_count")

            line = (
                f"- {name} | semantic_group={semantic_group} | table={table} | dtype={dtype}"
                f" | missing_rate={missing:.1%}"
                f" | non_null_count={non_null_count}"
                f" | desc={desc}"
            )
            if feature.get("mean") is not None:
                line += (
                    f" | mean={feature['mean']}"
                    f" | min={feature['min']}"
                    f" | max={feature['max']}"
                    f" | std={feature['std']}"
                    f" | var={feature.get('var')}"
                    f" | p05={feature.get('p05')}"
                    f" | p25={feature.get('p25')}"
                    f" | p50={feature.get('p50')}"
                    f" | p75={feature.get('p75')}"
                    f" | p95={feature.get('p95')}"
                )
            elif "unique_count" in feature:
                line += f" | unique_count={feature['unique_count']}"
            lines.append(line)

        features_text = "\n".join(lines)

        return f"""
You are a senior retail credit-risk feature-screening expert.

Context:
- You are reviewing metadata only, the way a strong credit-risk expert would review a variable pack before modeling.
- Your job is to create a broad, stability-aware candidate ranking for downstream model development.
- A downstream statistical selector will make the final redundancy pruning and final budget decision.

Task:
Produce a broad, stability-aware candidate ranking of up to {self.ranking_budget} features for a binary loan-default model.
Do not try to make the final statistical selection yourself. The downstream selector will refine this list for redundancy and final budget.

Priority criteria:
1. Stable out-of-time generalization.
2. Low missingness and broad coverage.
3. Interpretable credit-risk meaning.
4. Durable repayment behavior, indebtedness, leverage, exposure, utilization, capacity, delinquency, and customer history signals.
5. Avoid leakage-like artifacts, policy/process contamination, brittle operational proxies, and unstable sparse tail-dominated variables.
6. Control redundancy among near-duplicate aggregates, especially within the same semantic group, unless variants clearly add distinct information.
7. Prefer representatives that a credit-risk expert could defend in a stable scorecard review.

Rules:
1. Use only the feature names provided below.
2. You are seeing training-slice metadata only, not validation or OOT data.
3. Prefer broad semantic coverage rather than over-concentrating on one narrow family.
4. When two features look similar, prefer the less-missing, more interpretable, more operationally durable one.
5. Avoid selecting many duplicate aggregates from one business concept unless their summaries suggest meaningfully different information.
6. Return features in priority order, best first.
7. Do not invent new feature names.

Features:
{features_text}

Return ONLY valid JSON:
{{
  "selected_features": ["feature_1", "feature_2"],
  "reasoning_summary": "High-level reasoning.",
  "selection_principles": ["stability", "coverage", "redundancy control"],
  "feature_reasons": {{
    "feature_1": "Reason the feature belongs in the broad candidate pool."
  }}
}}
""".strip()

    def _build_metadata_signature(self, metadata: List[Dict], y: pd.Series | None) -> str:
        payload = {
            "feature_metadata": metadata,
            "n_features": len(metadata),
            "target_mean": round(float(pd.Series(y).mean()), 6) if y is not None else None,
            "max_missing_rate": self.max_missing_rate,
            "iv_filter_kwargs": self.iv_filter_kwargs,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_key_payload(self) -> dict[str, Any]:
        return {
            "metadata_signature": self.metadata_signature_,
            "scope": self.scope,
            "fold_id": self.fold_id,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash_,
            "ranking_budget": self.ranking_budget_config or {"max_shared_pool": self.ranking_budget},
            "shared_pool_size": self.shared_pool_size,
            "config_hash": self.config_hash,
        }

    def _cache_path(self) -> Path:
        if not self.shared_ranking_enabled:
            key = self._cache_key_payload()
            raw = json.dumps(key, sort_keys=True, ensure_ascii=True, default=str)
            signature = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return self.cache_dir / f"{signature}.json"

        safe_scope = self.scope.replace("/", "_").replace("\\", "_")
        key = self._cache_key_payload()
        raw = json.dumps(key, sort_keys=True, ensure_ascii=True, default=str)
        signature = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        fold_key = "final_dev" if self.scope == "final_dev" else f"fold_{self.fold_id}"
        return self.cache_dir / f"{safe_scope}_{fold_key}_{signature}.json"

    def _call_llm(self, prompt: str) -> dict:
        logger.info(
            "Calling LLM (%s) for ranking. Shared pool target: %s features.",
            self.model,
            self.ranking_budget,
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior retail credit-risk feature-screening expert "
                        "specializing in interpretable, stable variable selection."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "").strip()
        data = json.loads(content)
        raw_selected = data.get("selected_features", [])

        if not isinstance(raw_selected, list):
            raise ValueError("LLM response did not return selected_features as a list.")

        ordered_unique = list(dict.fromkeys(str(feature) for feature in raw_selected))
        raw_principles = data.get("selection_principles", [])
        if isinstance(raw_principles, list):
            selection_principles = list(
                dict.fromkeys(str(item).strip() for item in raw_principles if str(item).strip())
            )
        else:
            selection_principles = []

        usage = getattr(response, "usage", None)
        return {
            "selected_features": ordered_unique[: self.ranking_budget],
            "reasoning_summary": data.get("reasoning_summary", ""),
            "selection_principles": selection_principles,
            "feature_reasons": data.get("feature_reasons", {}),
            "request_model": self.model,
            "response_model": getattr(response, "model", self.model),
            "response_id": getattr(response, "id", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
            "raw_response": content,
        }

    def _write_artifacts(self, payload: dict, metadata: List[Dict], prompt: str) -> None:
        if self.artifact_dir is None:
            return

        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(metadata).to_csv(self.artifact_dir / "feature_metadata.csv", index=False)
        pd.DataFrame({"feature": payload["selected_features"]}).to_csv(
            self.artifact_dir / "selected_features.csv",
            index=False,
        )
        (self.artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (self.artifact_dir / "selection_payload.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_ranking_artifact(self, payload: dict) -> None:
        if self.ranking_artifact_dir is None or self.ranked_features_ is None:
            return

        self.ranking_artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.ranking_artifact_dir / "llm_rankings_summary.csv"
        reasons = payload.get("feature_reasons", {})
        if not isinstance(reasons, dict):
            reasons = {}

        rows = []
        for rank, feature in enumerate(self.ranked_features_, start=1):
            rows.append(
                {
                    "scope": self.scope,
                    "fold_id": self.fold_id if self.scope != "final_dev" else pd.NA,
                    "rank": rank,
                    "feature_name": feature,
                    "llm_reason": reasons.get(feature, payload.get("reasoning_summary", "")),
                    "metadata_signature": self.metadata_signature_,
                    "prompt_version": self.prompt_version,
                    "prompt_hash": self.prompt_hash_,
                    "config_hash": self.config_hash,
                    "shared_pool_size": self.shared_pool_size,
                    "selected_for_lr_top20": rank <= self.lr_feature_budget,
                    "selected_for_catboost_top40": rank <= self.catboost_feature_budget,
                    "candidate_for_lr_hybrid": rank <= self.lr_candidate_pool_budget,
                    "candidate_for_catboost_hybrid": rank <= self.catboost_candidate_pool_budget,
                    "cache_hit": self.cache_hit_,
                    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "request_model": payload.get("request_model", self.model),
                    "response_model": payload.get("response_model", self.model),
                    "prompt_tokens": payload.get("prompt_tokens"),
                    "completion_tokens": payload.get("completion_tokens"),
                    "total_tokens": payload.get("total_tokens"),
                }
            )

        new_df = pd.DataFrame(rows)
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["scope", "fold_id", "feature_name", "metadata_signature", "prompt_hash"],
                keep="last",
            )
        else:
            combined = new_df

        combined.sort_values(["scope", "fold_id", "rank"], na_position="last").to_csv(
            output_path,
            index=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if y is None:
            raise ValueError("LLMSelector requires target labels during fit.")

        candidate_X = X.copy()
        self.missing_filter_ = MissingRateFilter(max_missing_rate=self.max_missing_rate)
        candidate_X = self.missing_filter_.fit_transform(candidate_X, y)

        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self.missing_filter_.save_summary(self.artifact_dir / "missing_filter_summary.csv")

        if candidate_X.shape[1] == 0:
            raise ValueError("Missing-rate filter removed all candidate features for the LLM selector.")

        iv_filter = None
        if self.iv_filter_kwargs:
            iv_output_dir = None
            if self.artifact_dir is not None:
                iv_output_dir = self.artifact_dir / "iv_prefilter"

            iv_filter = IVWOEFilter(
                output_dir=str(iv_output_dir) if iv_output_dir is not None else None,
                **self.iv_filter_kwargs,
            )
            candidate_X = iv_filter.fit_transform(candidate_X, y)

        if candidate_X.shape[1] == 0:
            raise ValueError("IV prefilter removed all candidate features for the LLM selector.")

        metadata = self.feature_metadata or build_feature_metadata(
            X=candidate_X,
            description_csv_path=self.description_csv_path,
        )
        prompt = self._build_prompt(metadata)
        self.metadata_signature_ = self._build_metadata_signature(metadata, y)
        self.prompt_hash_ = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_file = self._cache_path()

        self.cache_file_ = cache_file
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            logger.info("Loading cached LLM ranking from %s", cache_file)
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            self.cache_hit_ = True
            self.llm_cache_hits_ = 1
        else:
            payload = self._call_llm(prompt)
            self.cache_hit_ = False
            self.llm_calls_made_ = 1
            payload.update(
                {
                    "metadata_signature": self.metadata_signature_,
                    "scope": self.scope,
                    "fold_id": self.fold_id,
                    "config_hash": self.config_hash,
                    "prompt_version": self.prompt_version,
                    "prompt_hash": self.prompt_hash_,
                    "ranking_budget": self.ranking_budget_config or {"max_shared_pool": self.ranking_budget},
                    "shared_pool_size": self.shared_pool_size,
                    "candidate_features": candidate_X.columns.tolist(),
                    "n_candidates": int(candidate_X.shape[1]),
                    "max_missing_rate": self.max_missing_rate,
                    "feature_metadata_rows": len(metadata),
                    "iv_selected_features": (
                        iv_filter.selected_features_ if iv_filter is not None else candidate_X.columns.tolist()
                    ),
                    "cache_key": self._cache_key_payload(),
                }
            )
            cache_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        raw_selected = list(dict.fromkeys(str(feature) for feature in payload.get("selected_features", [])))
        valid_ranking = [feature for feature in raw_selected if feature in candidate_X.columns]
        invalid_features = [feature for feature in raw_selected if feature not in candidate_X.columns]

        if len(valid_ranking) < self.ranking_budget:
            fallback_features = [
                feature for feature in candidate_X.columns.tolist() if feature not in valid_ranking
            ]
            valid_ranking.extend(fallback_features[: max(0, self.ranking_budget - len(valid_ranking))])

        valid_ranking = valid_ranking[: self.ranking_budget]
        if not valid_ranking:
            if iv_filter is not None and getattr(iv_filter, "iv_table_", None) is not None:
                valid_ranking = iv_filter.iv_table_.head(self.ranking_budget).index.tolist()
            else:
                valid_ranking = candidate_X.columns.tolist()[: self.ranking_budget]
            payload["fallback_reason"] = "llm_response_did_not_match_candidate_features"

        if invalid_features:
            payload["filtered_invalid_features"] = invalid_features

        self.ranked_features_ = valid_ranking
        self.selected_features = valid_ranking[: self.feature_budget]
        self.selected_features_ = self.selected_features
        self.selection_payload_ = payload

        self._write_artifacts(payload=payload, metadata=metadata, prompt=prompt)
        self._write_ranking_artifact(payload=payload)

        logger.info(
            "Successfully ranked %s features and selected top %s.",
            len(self.ranked_features_),
            len(self.selected_features),
        )
        return self

    def transform(self, X: pd.DataFrame):
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transform.")

        if self.missing_filter_ is not None:
            X = self.missing_filter_.transform(X)

        missing_features = [feature for feature in self.selected_features if feature not in X.columns]
        if missing_features:
            raise ValueError(
                f"Input is missing {len(missing_features)} selected features: {missing_features[:10]}"
            )

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        return self.fit(X, y).transform(X)
