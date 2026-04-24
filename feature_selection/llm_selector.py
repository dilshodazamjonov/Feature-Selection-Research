import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

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
    LLM-guided feature selector with fold-local metadata and audit artifacts.

    The selector optionally applies IV prefiltering inside ``fit`` so the LLM is
    only exposed to the current training slice. Selections are cached by a hash
    of the actual training metadata instead of a single global JSON file.
    """

    def __init__(
        self,
        description_csv_path: str,
        cache_dir: str = "outputs/llm_selector_cache",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_features: int = 50,
        max_missing_rate: float = 0.95,
        iv_filter_kwargs: Dict | None = None,
        feature_metadata: List[Dict] | None = None,
    ):
        self.description_csv_path = description_csv_path
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.temperature = temperature
        self.max_features = max_features
        self.max_missing_rate = max_missing_rate
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})
        self.feature_metadata = feature_metadata

        self.selected_features: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.artifact_dir: Path | None = None
        self.training_signature_: str | None = None
        self.cache_file_: Path | None = None
        self.selection_payload_: dict | None = None
        self._client: OpenAI | None = None
        self.missing_filter_: MissingRateFilter | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | os.PathLike) -> None:
        self.artifact_dir = Path(artifact_dir)

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
            missing = feature.get("missing_rate", 0.0)
            dtype = feature.get("dtype") or "unknown"
            non_null_count = feature.get("non_null_count")

            line = (
                f"- {name} | table={table} | dtype={dtype}"
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
                )
            elif "unique_count" in feature:
                line += f" | unique_count={feature['unique_count']}"
            lines.append(line)

        features_text = "\n".join(lines)

        return f"""
You are a senior credit risk modeling reviewer.

Task:
Select up to {self.max_features} features for a binary loan-default model.

Rules:
1. Use only the feature names provided below.
2. You are seeing summary metadata only, not row-level training data.
3. Prefer robust, plausibly causal signals over fragile or redundant ones.
4. Penalize high missingness unless the feature is clearly valuable.
5. Prefer simpler representatives among near-duplicate aggregates.
6. Return the selected features in priority order, best first.
7. Do not invent new feature names.

Features:
{features_text}

Return ONLY valid JSON:
{{
  "selected_features": ["feature_1", "feature_2"],
  "reasoning_summary": "Short explanation."
}}
""".strip()

    def _build_training_signature(self, metadata: List[Dict], y: pd.Series | None) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_features": self.max_features,
            "iv_filter_kwargs": self.iv_filter_kwargs,
            "feature_metadata": metadata,
            "n_features": len(metadata),
            "target_mean": round(float(pd.Series(y).mean()), 6) if y is not None else None,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, signature: str) -> Path:
        return self.cache_dir / f"{signature}.json"

    def _call_llm(self, prompt: str) -> dict:
        logger.info(
            "Calling LLM (%s) for selection. Target budget: %s features.",
            self.model,
            self.max_features,
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a credit risk expert specializing in feature engineering and model interpretability.",
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

        return {
            "selected_features": ordered_unique[: self.max_features],
            "reasoning_summary": data.get("reasoning_summary", ""),
            "request_model": self.model,
            "response_model": getattr(response, "model", self.model),
            "response_id": getattr(response, "id", None),
            "raw_response": content,
        }

    def _write_artifacts(
        self,
        payload: dict,
        metadata: List[Dict],
        prompt: str,
    ) -> None:
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
        signature = self._build_training_signature(metadata, y)
        cache_file = self._cache_path(signature)

        self.training_signature_ = signature
        self.cache_file_ = cache_file
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            logger.info("Loading cached LLM selection from %s", cache_file)
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            payload = self._call_llm(prompt)
            payload.update(
                {
                    "training_signature": signature,
                    "candidate_features": candidate_X.columns.tolist(),
                    "n_candidates": int(candidate_X.shape[1]),
                    "max_missing_rate": self.max_missing_rate,
                    "feature_metadata_rows": len(metadata),
                    "iv_selected_features": (
                        iv_filter.selected_features_ if iv_filter is not None else candidate_X.columns.tolist()
                    ),
                }
            )
            cache_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        valid_selected = [
            feature
            for feature in payload.get("selected_features", [])
            if feature in candidate_X.columns
        ]

        if not valid_selected:
            if iv_filter is not None:
                fallback_features = iv_filter.iv_table_.head(self.max_features).index.tolist()
            else:
                fallback_features = candidate_X.columns.tolist()[: self.max_features]
            payload["selected_features"] = fallback_features
            payload["fallback_reason"] = "llm_response_did_not_match_candidate_features"
            valid_selected = fallback_features

        self.selected_features = valid_selected
        self.selected_features_ = valid_selected
        self.selection_payload_ = payload

        self._write_artifacts(payload=payload, metadata=metadata, prompt=prompt)

        logger.info("Successfully selected %s features.", len(self.selected_features))
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
