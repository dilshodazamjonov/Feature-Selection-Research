"""LLM pre-screening under the definition-only information contract.

The language model reads the variable book only: feature names, definitions, source
lineage, aggregation operations and observation windows. It never receives a data row,
an outcome, a validation statistic or a held-out value. One prompt template
(:mod:`credit_risk_fs.selectors.llm_prompt`) serves all three datasets.

The candidate list is fixed upstream (the screened candidate set) and is the same in
every call. A response is accepted only when it holds exactly the requested number of
distinct names copied verbatim from the candidate records; a failing response is
re-requested up to ``max_rerequests`` times and no fallback ranking is used. Accepted
responses are cached, and the cached response, not a fresh call, is the reproducible
artefact.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import dotenv
import pandas as pd

from credit_risk_fs.feature_metadata.builder import infer_semantic_group
from credit_risk_fs.feature_metadata.definitions import generic_definition_records
from credit_risk_fs.selectors.base import SelectedFeaturesMixin, select_feature_frame
from credit_risk_fs.selectors.llm_prompt import (
    PROMPT_VERSION,
    RankingResponseRejected,
    build_request,
    render_prompt,
    request_sha256,
    response_format,
    validate_ranking_response,
)
from credit_risk_fs.utils.logging import setup_logging

dotenv.load_dotenv()

logger = setup_logging("llm_selector", level=logging.INFO)

DEFAULT_MODEL_SNAPSHOT = "gpt-4.1-mini-2025-04-14"


class LLMRankingFailed(RuntimeError):
    """Raised when no response satisfies the output contract; there is no fallback."""


def request_ranking(
    *,
    prompt: str,
    candidate_names: Sequence[str],
    ranking_budget: int,
    model: str,
    temperature: float,
    max_rerequests: int = 3,
    client: Any | None = None,
    attempt_recorder: Callable[[Mapping[str, Any]], None] | None = None,
    expected_response_model: str | None = None,
) -> dict[str, Any]:
    """Send one ranking request; re-request a failing response up to ``max_rerequests`` times."""

    request = build_request(model=model, temperature=temperature, prompt=prompt)
    if client is None:
        from openai import OpenAI

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set; it is required for a fresh LLM ranking call.")
        client = OpenAI(api_key=key)
    errors: list[str] = []
    for attempt in range(1, int(max_rerequests) + 2):
        response = client.chat.completions.create(
            model=request["model"],
            temperature=request["temperature"],
            messages=request["messages"],
            response_format=request["response_format"],
        )
        content = (response.choices[0].message.content or "").strip()
        error: str | None = None
        ranking: list[str] | None = None
        try:
            data = json.loads(content)
            ranking = validate_ranking_response(
                data, candidate_names=candidate_names, ranking_budget=ranking_budget
            )
            response_model = str(getattr(response, "model", ""))
            if expected_response_model is not None and response_model != expected_response_model:
                raise RankingResponseRejected(
                    f"response model {response_model!r} differs from {expected_response_model!r}"
                )
        except (json.JSONDecodeError, RankingResponseRejected) as exc:
            error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            errors.append(error)
            ranking = None
        if attempt_recorder is not None:
            attempt_recorder(
                {
                    "attempt": attempt,
                    "response_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "response": {"raw_content": content},
                    "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "validation_error": error,
                    "valid": ranking is not None,
                }
            )
        if ranking is not None:
            usage = getattr(response, "usage", None)
            return {
                "selected_features": ranking,
                "reasoning_summary": str(data.get("reasoning_summary", "")),
                "selection_principles": [str(item) for item in data.get("selection_principles", [])],
                "request_model": request["model"],
                "response_model": str(getattr(response, "model", "")),
                "response_id": getattr(response, "id", None),
                "temperature": request["temperature"],
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "request_sha256": request_sha256(request),
                "attempts": attempt,
                "rejected_attempts": errors,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
                "raw_response": content,
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        logger.warning("LLM ranking response rejected (%s)", error)
    raise LLMRankingFailed(
        f"no response satisfied the output contract after {int(max_rerequests) + 1} attempts: {errors}"
    )


class LLMSelector(SelectedFeaturesMixin):
    """Fold-local LLM ranking selector reading feature definitions only."""

    TARGET_FREE_METADATA_FIELDS = frozenset(
        {
            "name",
            "source_family",
            "source_table",
            "original_feature",
            "depth",
            "aggregation",
            "dtype",
            "logical_type",
            "approved_definition",
            "rendered_description",
            "description_sha256",
        }
    )

    def __init__(
        self,
        description_csv_path: str | None = None,
        cache_dir: str = "results/_llm_rankings_cache",
        model: str = DEFAULT_MODEL_SNAPSHOT,
        temperature: float = 0.0,
        max_features: int = 100,
        ranking_budget: int | None = None,
        feature_budget: int | None = None,
        shared_ranking_enabled: bool = True,
        config_hash: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        ranking_budget_config: Dict[str, int] | None = None,
        shared_pool_size: int | None = None,
        lr_feature_budget: int = 20,
        catboost_feature_budget: int = 40,
        lr_candidate_pool_budget: int = 60,
        catboost_candidate_pool_budget: int = 100,
        feature_records: Sequence[Mapping[str, Any]] | None = None,
        max_rerequests: int = 3,
        max_missing_rate: float | None = None,
        iv_filter_kwargs: Dict | None = None,
        feature_metadata: List[Dict] | None = None,
    ):
        if iv_filter_kwargs:
            raise ValueError(
                "the LLM selector reads feature definitions only; an IV/WOE prefilter is not permitted"
            )
        if max_missing_rate is not None:
            raise ValueError(
                "the LLM selector receives the fixed candidate set; a per-call missingness screen is not permitted"
            )
        if feature_metadata is not None:
            raise ValueError("data-derived feature metadata is not permitted; supply feature_records")
        self.description_csv_path = description_csv_path
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.temperature = float(temperature)
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
        self.feature_records = None if feature_records is None else [dict(item) for item in feature_records]
        self.max_rerequests = int(max_rerequests)

        self.ranked_features_: list[str] | None = None
        self.selected_features_ = None
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

    # ------------------------------------------------------------ records
    def _records_for(self, candidates: Sequence[str]) -> list[dict[str, Any]]:
        if self.feature_records is not None:
            by_name = {str(record["name"]): record for record in self.feature_records}
            missing = [name for name in candidates if name not in by_name]
            if missing:
                raise ValueError(f"feature records do not cover the candidates: {missing[:5]}")
            return [dict(by_name[name]) for name in candidates]
        return generic_definition_records(candidates, description_csv_path=self.description_csv_path)

    # ------------------------------------------------------------ target-free entry points
    def _validate_target_free_metadata(
        self,
        metadata: Sequence[Mapping[str, Any]],
        expected_features: Sequence[str],
    ) -> list[dict[str, Any]]:
        expected = [str(value) for value in expected_features]
        if len(expected) != len(set(expected)):
            raise ValueError("target-free candidate universe contains duplicate names")
        records: list[dict[str, Any]] = []
        for index, raw in enumerate(metadata):
            record = dict(raw)
            unknown = set(record) - self.TARGET_FREE_METADATA_FIELDS
            if unknown:
                raise ValueError(f"target-free metadata contains non-authorized fields: {sorted(unknown)}")
            name = str(record.get("name"))
            if index >= len(expected) or name != expected[index]:
                raise ValueError("target-free metadata order differs from candidate universe")
            records.append(record)
        if len(records) != len(expected):
            raise ValueError("target-free metadata does not cover the complete candidate universe")
        return records

    def _records_from_target_free_metadata(
        self,
        metadata: Sequence[Mapping[str, Any]],
        expected_features: Sequence[str],
    ) -> list[dict[str, Any]]:
        converted = []
        for record in self._validate_target_free_metadata(metadata, expected_features):
            family = str(record.get("source_family", ""))
            original = str(record.get("original_feature", record["name"]))
            aggregation = str(record.get("aggregation", "none"))
            description = " ".join(str(record.get("approved_definition", "")).split())
            if aggregation in {"identity", "identity_after_family_prefix", "raw", "none"}:
                lineage = f"{family}.{original}"
            else:
                lineage = f"{aggregation}({family}.{original})"
            converted.append(
                {
                    "name": str(record["name"]),
                    "description": description,
                    "source_family": family,
                    "lineage": lineage,
                    "semantic_group": infer_semantic_group(
                        str(record["name"]), description=description, table=family
                    ),
                }
            )
        return converted

    def build_target_free_prompt(
        self,
        metadata: Sequence[Mapping[str, Any]],
        *,
        expected_features: Sequence[str],
    ) -> str:
        records = self._records_from_target_free_metadata(metadata, expected_features)
        return render_prompt(ranking_budget=self.ranking_budget, records=records)

    def target_free_response_format(self) -> dict[str, Any]:
        return response_format()

    def rank_target_free(
        self,
        metadata: Sequence[Mapping[str, Any]],
        *,
        expected_features: Sequence[str],
        expected_response_model: str,
        attempt_recorder: Callable[[Mapping[str, Any]], None] | None = None,
        maximum_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Rank pre-rendered target-free records with the unified prompt; no fallback."""

        prompt = self.build_target_free_prompt(metadata, expected_features=expected_features)
        rerequests = self.max_rerequests if maximum_attempts is None else max(0, int(maximum_attempts) - 1)
        payload = request_ranking(
            prompt=prompt,
            candidate_names=[str(value) for value in expected_features],
            ranking_budget=self.ranking_budget,
            model=self.model,
            temperature=self.temperature,
            max_rerequests=rerequests,
            client=getattr(self, "_client", None),
            attempt_recorder=attempt_recorder,
            expected_response_model=expected_response_model,
        )
        payload["application_attempt"] = payload["attempts"]
        payload["fallback_used"] = False
        return payload

    # ------------------------------------------------------------ cache
    def _cache_path(self, request_digest: str) -> Path:
        safe_scope = self.scope.replace("/", "_").replace("\\", "_")
        fold_key = "final_dev" if self.scope == "final_dev" else f"fold_{self.fold_id}"
        return self.cache_dir / f"{safe_scope}_{fold_key}_{request_digest}.json"

    def _write_ranking_artifact(self, payload: dict) -> None:
        if self.ranking_artifact_dir is None or self.ranked_features_ is None:
            return
        self.ranking_artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.ranking_artifact_dir / "llm_rankings_summary.csv"
        rows = [
            {
                "scope": self.scope,
                "fold_id": self.fold_id if self.scope != "final_dev" else pd.NA,
                "rank": rank,
                "feature_name": feature,
                "metadata_signature": self.metadata_signature_,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash_,
                "config_hash": self.config_hash,
                "shared_pool_size": self.shared_pool_size,
                "feature_budget": self.feature_budget,
                "cache_hit": self.cache_hit_,
                "cache_file_name": None if self.cache_file_ is None else self.cache_file_.name,
                "request_model": payload.get("request_model", self.model),
                "response_model": payload.get("response_model"),
                "response_id": payload.get("response_id"),
                "temperature": payload.get("temperature", self.temperature),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "total_tokens": payload.get("total_tokens"),
            }
            for rank, feature in enumerate(self.ranked_features_, start=1)
        ]
        new_df = pd.DataFrame(rows)
        if output_path.exists():
            combined = pd.concat([pd.read_csv(output_path), new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["scope", "fold_id", "feature_name", "metadata_signature", "prompt_hash"],
                keep="last",
            )
        else:
            combined = new_df
        combined.sort_values(["scope", "fold_id", "rank"], na_position="last").to_csv(output_path, index=False)

    # ------------------------------------------------------------ fit / transform
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Rank the columns of ``X`` (the fixed candidate set) from their definitions.

        ``y`` is accepted for interface compatibility and is never read.
        """

        candidates = [str(column) for column in X.columns]
        if len(candidates) < self.ranking_budget:
            raise ValueError(
                f"the candidate set holds {len(candidates)} features, fewer than the ranking budget {self.ranking_budget}"
            )
        records = self._records_for(candidates)
        prompt = render_prompt(ranking_budget=self.ranking_budget, records=records)
        request = build_request(model=self.model, temperature=self.temperature, prompt=prompt)
        digest = request_sha256(request)
        self.prompt_hash_ = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.metadata_signature_ = hashlib.sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_file = self._cache_path(digest)
        self.cache_file_ = cache_file
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        payload = None
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            ranking = validate_ranking_response(
                {"status": "ok", "selected_features": payload.get("selected_features")},
                candidate_names=candidates,
                ranking_budget=self.ranking_budget,
            )
            self.cache_hit_ = True
            self.llm_cache_hits_ = 1
        else:
            payload = request_ranking(
                prompt=prompt,
                candidate_names=candidates,
                ranking_budget=self.ranking_budget,
                model=self.model,
                temperature=self.temperature,
                max_rerequests=self.max_rerequests,
            )
            payload.update(
                {
                    "scope": self.scope,
                    "fold_id": self.fold_id,
                    "candidate_features": candidates,
                    "feature_records": records,
                    "config_hash": self.config_hash,
                }
            )
            tmp = cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_file)
            ranking = list(payload["selected_features"])
            self.cache_hit_ = False
            self.llm_calls_made_ = 1

        self.ranked_features_ = ranking
        self.selected_features_ = ranking[: self.feature_budget]
        self.selection_payload_ = payload
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            (self.artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            pd.DataFrame(records).to_csv(self.artifact_dir / "feature_records.csv", index=False)
            pd.DataFrame({"feature": self.selected_features_}).to_csv(
                self.artifact_dir / "selected_features.csv", index=False
            )
        self._write_ranking_artifact(payload)
        logger.info(
            "Ranked %s candidates and selected the top %s.", len(candidates), len(self.selected_features_)
        )
        return self

    def transform(self, X: pd.DataFrame):
        return select_feature_frame(X, self.selected_features_, selector_name=self.__class__.__name__)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self.fit(X, y).transform(X)


__all__ = ["DEFAULT_MODEL_SNAPSHOT", "LLMRankingFailed", "LLMSelector", "request_ranking"]
