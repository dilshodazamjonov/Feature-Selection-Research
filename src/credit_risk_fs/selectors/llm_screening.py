import hashlib
import inspect
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import dotenv
import pandas as pd
from openai import OpenAI

from iv_woe_filter import IVWOEFilter
from credit_risk_fs.feature_metadata.builder import build_feature_metadata
from credit_risk_fs.preprocessing.missingness import MissingRateFilter
from credit_risk_fs.selectors.base import SelectedFeaturesMixin, select_feature_frame
from credit_risk_fs.utils.logging import setup_logging

dotenv.load_dotenv()

logger = setup_logging("llm_selector", level=logging.INFO)


class LLMSelector(SelectedFeaturesMixin):
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
        self._client: OpenAI | None = None
        self.missing_filter_: MissingRateFilter | None = None
        self.select_before_preprocessing = True

    def _prepare_candidate_frame_for_iv(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared = X.copy()
        categorical_cols = prepared.select_dtypes(include=["category"]).columns.tolist()
        for column in categorical_cols:
            prepared[column] = prepared[column].astype("object")
        return prepared

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
  "reasoning_summary": "One concise high-level reason.",
  "selection_principles": ["stability", "coverage", "redundancy control"],
  "feature_reasons": {{}}
}}

Keep the response compact. Do not include per-feature explanations unless absolutely necessary.
""".strip()

    TARGET_FREE_SYSTEM_MESSAGE = (
        "You are a senior retail credit-risk feature-screening expert "
        "specializing in interpretable, stable variable selection."
    )
    TARGET_FREE_RETRY_SUFFIX = (
        "CRITICAL RETRY INSTRUCTION: the previous response failed the frozen JSON "
        "or candidate-coverage contract. Fill exactly the requested number of slots in "
        "selected_features with distinct names copied verbatim from the candidate list. "
        "Candidates outside those slots are intentionally not selected. Do not return "
        "extra names and do not invent, normalize, replace, or duplicate a selected name."
    )
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

    def _validate_target_free_metadata(
        self,
        metadata: Sequence[Mapping[str, Any]],
        expected_features: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Validate the outcome-independent Prompt-16 description boundary.

        This is intentionally separate from ``fit``.  It never receives a feature
        matrix or target and therefore cannot place fold statistics in the prompt.
        """

        expected = [str(value) for value in expected_features]
        if len(expected) != len(set(expected)):
            raise ValueError("target-free candidate universe contains duplicate names")
        records: list[dict[str, Any]] = []
        for index, raw in enumerate(metadata):
            record = dict(raw)
            unknown = set(record) - self.TARGET_FREE_METADATA_FIELDS
            if unknown:
                raise ValueError(
                    "target-free metadata contains non-authorized fields: "
                    f"{sorted(unknown)}"
                )
            missing = self.TARGET_FREE_METADATA_FIELDS - set(record)
            if missing:
                raise ValueError(
                    "target-free metadata is missing required fields: "
                    f"{sorted(missing)}"
                )
            name = str(record["name"])
            if index >= len(expected) or name != expected[index]:
                raise ValueError("target-free metadata order differs from candidate universe")
            rendered = str(record["rendered_description"])
            observed_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if observed_hash != str(record["description_sha256"]):
                raise ValueError(f"rendered feature-description hash mismatch: {name}")
            records.append(record)
        if len(records) != len(expected):
            raise ValueError(
                "target-free metadata does not cover the complete candidate universe"
            )
        return records

    def build_target_free_prompt(
        self,
        metadata: Sequence[Mapping[str, Any]],
        *,
        expected_features: Sequence[str],
    ) -> str:
        """Render the stability_expert_v3 ranking prompt without data statistics."""

        records = self._validate_target_free_metadata(metadata, expected_features)
        features_text = "\n".join(
            str(record["rendered_description"]) for record in records
        )
        return f"""
You are a senior retail credit-risk feature-screening expert.

Context:
- You are reviewing outcome-independent feature definitions and adapter lineage only, the way a strong credit-risk expert would review a variable pack before modeling.
- Your job is to create a broad, stability-aware candidate ranking for downstream model development.
- A downstream statistical selector may perform fold-local supervised stability fitting and final redundancy pruning.

Task:
Produce a broad, stability-aware candidate ranking of exactly {self.ranking_budget} features for a binary loan-default model.
Do not try to make the final statistical selection yourself. Downstream consumers will apply the frozen model-specific budgets.

Priority criteria:
1. Stable out-of-time generalization.
2. Interpretable credit-risk meaning grounded only in the supplied definitions and lineage.
3. Durable repayment behavior, indebtedness, leverage, exposure, utilization, capacity, delinquency, and customer-history signals.
4. Avoid leakage-like artifacts, policy/process contamination, and brittle operational proxies identifiable from definitions alone.
5. Control redundancy among near-duplicate aggregates, especially within the same source family, unless variants clearly add distinct information.
6. Prefer representatives that a credit-risk expert could defend in a stable scorecard review.

Rules:
1. Use only the feature names provided below and copy every selected name verbatim.
2. You are seeing no rows, targets, split statistics, validation information, OOT information, performance, drift, missingness rates, IV, correlation, mutual information, SHAP, or model importance.
3. Prefer broad semantic coverage rather than over-concentrating on one narrow family.
4. Return exactly {self.ranking_budget} distinct features in priority order, best first.
5. Every candidate not placed in the {self.ranking_budget}-item selected_features array is intentionally not selected.
6. Do not return more or fewer than {self.ranking_budget} names. Do not invent, normalize, replace, or duplicate a selected name.

Features:
{features_text}

Return ONLY valid JSON:
{{
  "selected_features": ["feature_1", "feature_2"],
  "reasoning_summary": "One concise high-level reason.",
  "selection_principles": ["stability", "coverage", "redundancy control"],
  "feature_reasons": {{}}
}}

Keep the response compact. Do not include per-feature explanations unless absolutely necessary.
""".strip()

    def _normalize_target_free_response(
        self,
        data: Any,
        *,
        candidate_features: Sequence[str],
        response: Any,
        content: str,
        expected_response_model: str,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object")
        raw_selected = data.get("selected_features")
        if not isinstance(raw_selected, list):
            raise ValueError("LLM response did not return selected_features as a list")
        if len(raw_selected) != self.ranking_budget:
            raise ValueError(
                "LLM response selected_features count mismatch: "
                f"{len(raw_selected)} != {self.ranking_budget}"
            )
        if any(not isinstance(value, str) for value in raw_selected):
            raise ValueError("LLM response contains a non-string feature name")
        selected = list(raw_selected)
        if len(selected) != len(set(selected)):
            raise ValueError("LLM response contains duplicate feature names")
        candidates = set(map(str, candidate_features))
        unknown = [value for value in selected if value not in candidates]
        if unknown:
            raise ValueError(f"LLM response contains unknown feature names: {unknown[:10]}")
        response_model = str(getattr(response, "model", ""))
        if response_model != expected_response_model:
            raise ValueError(
                "LLM response model identity mismatch: "
                f"{response_model!r} != {expected_response_model!r}"
            )
        raw_principles = data.get("selection_principles", [])
        principles = (
            [str(value).strip() for value in raw_principles if str(value).strip()]
            if isinstance(raw_principles, list)
            else []
        )
        usage = getattr(response, "usage", None)
        return {
            "selected_features": selected,
            "reasoning_summary": str(data.get("reasoning_summary", "")),
            "selection_principles": principles,
            "feature_reasons": (
                data.get("feature_reasons", {})
                if isinstance(data.get("feature_reasons", {}), dict)
                else {}
            ),
            "provider": "openai",
            "request_model": self.model,
            "response_model": response_model,
            "response_id": getattr(response, "id", None),
            "temperature": self.temperature,
            "seed": None,
            "prompt_tokens": (
                getattr(usage, "prompt_tokens", None) if usage is not None else None
            ),
            "completion_tokens": (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            ),
            "total_tokens": (
                getattr(usage, "total_tokens", None) if usage is not None else None
            ),
            "raw_response": content,
            "fallback_used": False,
            "candidate_coverage": {
                "input_candidates": len(candidates),
                "ranked_features": len(selected),
                "unknown_features": 0,
                "duplicate_features": 0,
                "missing_required_rank_positions": 0,
            },
        }

    def target_free_response_format(self) -> dict[str, Any]:
        """Return the strict schema for one exact-size target-free ranking."""

        budget = int(self.ranking_budget)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "target_free_feature_ranking",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_features": {
                            "type": "array",
                            "description": (
                                f"Exactly {budget} distinct candidate feature names in "
                                "priority order."
                            ),
                            "items": {"type": "string"},
                            "minItems": budget,
                            "maxItems": budget,
                        },
                        "reasoning_summary": {"type": "string"},
                        "selection_principles": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "feature_reasons": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "selected_features",
                        "reasoning_summary",
                        "selection_principles",
                        "feature_reasons",
                    ],
                    "additionalProperties": False,
                },
            },
        }

    def rank_target_free(
        self,
        metadata: Sequence[Mapping[str, Any]],
        *,
        expected_features: Sequence[str],
        expected_response_model: str,
        attempt_recorder: Callable[[Mapping[str, Any]], None] | None = None,
        maximum_attempts: int = 3,
    ) -> dict[str, Any]:
        """Request one strict, outcome-independent ranking with no fallback.

        The OpenAI SDK retains its historical transport retry behavior.  These
        application attempts are only for malformed JSON or a failed strict
        candidate-coverage/model-identity check.
        """

        if maximum_attempts != 3:
            raise ValueError("the frozen target-free ranking contract requires 3 attempts")
        prompt = self.build_target_free_prompt(
            metadata, expected_features=expected_features
        )
        client = self._get_client()
        response_format = self.target_free_response_format()
        validation_errors: list[str] = []
        for attempt in range(1, maximum_attempts + 1):
            user_prompt = (
                prompt
                if attempt == 1
                else f"{prompt}\n\n{self.TARGET_FREE_RETRY_SUFFIX}"
            )
            request = {
                "provider": "openai",
                "endpoint": "chat.completions.create",
                "model": self.model,
                "temperature": self.temperature,
                "seed": None,
                "messages": [
                    {"role": "system", "content": self.TARGET_FREE_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": response_format,
            }
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=request["messages"],
                    response_format=request["response_format"],
                )
            except Exception as exc:
                if attempt_recorder is not None:
                    attempt_recorder(
                        {
                            "attempt": attempt,
                            "request": request,
                            "request_sha256": hashlib.sha256(
                                json.dumps(
                                    request,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ).encode("utf-8")
                            ).hexdigest(),
                            "response": {
                                "id": None,
                                "model": None,
                                "raw_content": "",
                                "raw_content_sha256": hashlib.sha256(b"").hexdigest(),
                                "transport_error": {
                                    "class": type(exc).__name__,
                                    "message": str(exc),
                                },
                            },
                            "validation_error": (
                                f"transport_error: {type(exc).__name__}: {exc}"
                            ),
                            "valid": False,
                        }
                    )
                raise
            content = (response.choices[0].message.content or "").strip()
            error: str | None = None
            payload: dict[str, Any] | None = None
            try:
                data = json.loads(content)
                payload = self._normalize_target_free_response(
                    data,
                    candidate_features=expected_features,
                    response=response,
                    content=content,
                    expected_response_model=expected_response_model,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                validation_errors.append(f"attempt={attempt}: {error}")
            attempt_record = {
                "attempt": attempt,
                "request": request,
                "request_sha256": hashlib.sha256(
                    json.dumps(
                        request,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "response": {
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", None),
                    "raw_content": content,
                    "raw_content_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                },
                "validation_error": error,
                "valid": payload is not None,
            }
            if attempt_recorder is not None:
                attempt_recorder(attempt_record)
            if payload is not None:
                payload["application_attempt"] = attempt
                payload["validation_errors_before_success"] = validation_errors
                payload["prompt_sha256"] = hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest()
                return payload
        raise ValueError(
            "LLM response failed the strict target-free ranking contract after "
            f"{maximum_attempts} attempts: {validation_errors}"
        )

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
        payload = {
            "metadata_signature": self.metadata_signature_,
            "scope": self.scope,
            "fold_id": self.fold_id,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash_,
            "ranking_budget": self.ranking_budget_config or {"max_shared_pool": self.ranking_budget},
            "shared_pool_size": self.shared_pool_size,
            "config_hash": self.config_hash,
            "model": self.model,
            "temperature": self.temperature,
            "description_csv_path": str(self.description_csv_path),
        }
        if not self.shared_ranking_enabled:
            payload["feature_budget"] = self.feature_budget
        return payload

    def _cache_key_hash(self) -> str:
        raw = json.dumps(self._cache_key_payload(), sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

    def _normalize_llm_response(self, data: dict, response: Any, content: str) -> dict:
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

    def _fallback_payload(
        self,
        *,
        candidate_features: list[str],
        errors: list[str],
        raw_response: str,
    ) -> dict:
        logger.error(
            "LLM ranking response could not be parsed after retries; using deterministic fallback ranking."
        )
        return {
            "selected_features": candidate_features[: self.ranking_budget],
            "reasoning_summary": (
                "Deterministic fallback ranking used because the LLM returned malformed JSON."
            ),
            "selection_principles": ["deterministic_fallback_after_llm_json_error"],
            "feature_reasons": {},
            "request_model": self.model,
            "response_model": self.model,
            "response_id": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "raw_response": raw_response,
            "fallback_reason": "llm_json_parse_failed_after_retries",
            "llm_response_parse_errors": errors,
        }

    def _call_llm(self, prompt: str, candidate_features: list[str] | None = None) -> dict:
        logger.info(
            "Calling LLM (%s) for ranking. Shared pool target: %s features.",
            self.model,
            self.ranking_budget,
        )

        client = self._get_client()
        system_message = (
            "You are a senior retail credit-risk feature-screening expert "
            "specializing in interpretable, stable variable selection."
        )
        errors: list[str] = []
        last_content = ""
        for attempt in range(1, 4):
            user_prompt = prompt
            if attempt > 1:
                user_prompt = (
                    f"{prompt}\n\n"
                    "CRITICAL RETRY INSTRUCTION: the previous response was not parseable JSON. "
                    "Return compact valid JSON only. Include selected_features, a short "
                    "reasoning_summary, selection_principles, and set feature_reasons to {}."
                )
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            last_content = content
            try:
                data = json.loads(content)
                return self._normalize_llm_response(data, response, content)
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"attempt={attempt}: {type(exc).__name__}: {exc}"
                errors.append(error)
                logger.warning("Invalid LLM JSON ranking response (%s).", error)

        if candidate_features:
            return self._fallback_payload(
                candidate_features=candidate_features,
                errors=errors,
                raw_response=last_content,
            )
        raise ValueError(f"LLM response did not produce parseable ranking JSON: {errors}")

    def _call_llm_for_ranking(self, prompt: str, candidate_features: list[str]) -> dict:
        """Call the LLM hook while preserving compatibility with old test doubles."""
        signature = inspect.signature(self._call_llm)
        supports_candidate_features = (
            "candidate_features" in signature.parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        )
        if supports_candidate_features:
            return self._call_llm(prompt, candidate_features=candidate_features)
        return self._call_llm(prompt)

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
        raw_response = payload.get("raw_response")
        if raw_response:
            try:
                raw_payload = json.loads(str(raw_response))
            except json.JSONDecodeError:
                raw_payload = {"raw_response": str(raw_response)}
            (self.artifact_dir / "raw_llm_response.json").write_text(
                json.dumps(raw_payload, indent=2, ensure_ascii=False),
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
                    "feature_budget": self.feature_budget,
                    "selected_for_lr_top20": rank <= self.lr_feature_budget,
                    "selected_for_catboost_top40": rank <= self.catboost_feature_budget,
                    "candidate_for_lr_hybrid": rank <= self.lr_candidate_pool_budget,
                    "candidate_for_catboost_hybrid": rank <= self.catboost_candidate_pool_budget,
                    "cache_hit": self.cache_hit_,
                    "cache_key_hash": payload.get("cache_key_hash"),
                    "cache_file_name": payload.get("cache_file_name"),
                    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "request_model": payload.get("request_model", self.model),
                    "response_model": payload.get("response_model", self.model),
                    "response_id": payload.get("response_id"),
                    "temperature": payload.get("temperature", self.temperature),
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

            candidate_X = self._prepare_candidate_frame_for_iv(candidate_X)
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
        cache_key_hash = self._cache_key_hash()

        payload = None
        if cache_file.exists():
            logger.info("Loading cached LLM ranking from %s", cache_file)
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                invalid_path = cache_file.with_name(
                    f"{cache_file.stem}.invalid_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{cache_file.suffix}"
                )
                cache_file.replace(invalid_path)
                logger.warning(
                    "Moved invalid LLM cache file to %s after JSON parse failure: %s",
                    invalid_path,
                    exc,
                )

        if payload is not None:
            self.cache_hit_ = True
            self.llm_cache_hits_ = 1
            payload.setdefault("metadata_signature", self.metadata_signature_)
            payload.setdefault("scope", self.scope)
            payload.setdefault("fold_id", self.fold_id)
            payload.setdefault("config_hash", self.config_hash)
            payload.setdefault("prompt_version", self.prompt_version)
            payload.setdefault("prompt_hash", self.prompt_hash_)
            payload.setdefault(
                "ranking_budget",
                self.ranking_budget_config or {"max_shared_pool": self.ranking_budget},
            )
            payload.setdefault("shared_pool_size", self.shared_pool_size)
            payload.setdefault("feature_budget", self.feature_budget)
            payload.setdefault("temperature", self.temperature)
            payload.setdefault("cache_key", self._cache_key_payload())
            payload.setdefault("cache_key_hash", cache_key_hash)
            payload.setdefault("cache_file_name", cache_file.name)
        else:
            payload = self._call_llm_for_ranking(prompt, candidate_X.columns.tolist())
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
                    "feature_budget": self.feature_budget,
                    "temperature": self.temperature,
                    "candidate_features": candidate_X.columns.tolist(),
                    "n_candidates": int(candidate_X.shape[1]),
                    "max_missing_rate": self.max_missing_rate,
                    "feature_metadata_rows": len(metadata),
                    "iv_selected_features": (
                        iv_filter.selected_features_ if iv_filter is not None else candidate_X.columns.tolist()
                    ),
                    "cache_key": self._cache_key_payload(),
                    "cache_key_hash": cache_key_hash,
                    "cache_file_name": cache_file.name,
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
        self.selected_features_ = valid_ranking[: self.feature_budget]
        self.selection_payload_ = payload

        self._write_artifacts(payload=payload, metadata=metadata, prompt=prompt)
        self._write_ranking_artifact(payload=payload)

        logger.info(
            "Successfully ranked %s features and selected top %s.",
            len(self.ranked_features_),
            len(self.selected_features_),
        )
        return self

    def transform(self, X: pd.DataFrame):
        if self.missing_filter_ is not None:
            X = self.missing_filter_.transform(X)
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        return self.fit(X, y).transform(X)



# Semantically meaningful and statistically stable 
# apply clip for the feature selection 
