import os
import json
import logging
from typing import List, Dict
import pandas as pd
from openai import OpenAI
from utils.logging_config import setup_logging
import dotenv

dotenv.load_dotenv()

logger = setup_logging("llm_selector", level=logging.INFO)

class LLMSelector:
    """
    Feature selection using an LLM based on domain knowledge and data distributions.
    Optimized for research comparisons with fixed feature budgets.
    """

    def __init__(
        self,
        feature_metadata: List[Dict],
        cache_path: str = "llm_cache/llm_selected_features.json",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_features: int = 50,
    ):
        """
        Args:
            feature_metadata: List of feature dictionaries from build_feature_metadata.
            cache_path: Path to save/load selected features.
            model: LLM model name.
            temperature: Sampling temperature (0 for reproducibility).
            max_features: The target number of features the LLM should attempt to select.
        """
        self.feature_metadata = feature_metadata
        self.cache_path = cache_path
        self.model = model
        self.temperature = temperature
        self.max_features = max_features
        self.selected_features = None

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key not found. Pass it to the constructor or set OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=key)

    def _build_prompt(self) -> str:
        """
        Builds a compact, token-efficient prompt for the LLM.
        """
        lines = []
        for f in self.feature_metadata:
            name = f["name"]
            desc = f.get("description", "No description")
            table = f.get("table", "Unknown")
            missing = f.get("missing_rate", 0)
            
            # Base info with rounded missing rate
            info = f"- {name} (Table: {table}): {desc} | Missing: {missing:.1%}"
            
            # Add stats only if they exist (Numeric)
            if "mean" in f:
                info += f" | Stats: [Mean: {f['mean']:.2f}, Range: {f['min']:.2f} to {f['max']:.2f}]"
            # Add unique counts (Categorical)
            elif "unique_count" in f:
                info += f" | Unique: {f['unique_count']}"
                
            lines.append(info)

        features_text = "\n".join(lines)

        prompt = f"""
You are a senior Credit Risk Expert. 

Task:
Select the top {self.max_features} features most relevant for predicting loan default risk. 

Criteria:
1. Prioritize causal financial drivers (e.g., payment behavior, debt-to-income).
2. Penalize features with high missing rates (>60%) unless theoretically vital.
3. Identify and remove redundant features (keep the one with best data quality).
4. Use the provided statistics to judge if the feature range makes sense.

Features:
{features_text}

Return ONLY valid JSON:
{{
    "selected_features": ["feature1", "feature2", ...],
    "reasoning_summary": "Explain why these specific features were chosen over others."
}}
"""
        return prompt

    def _call_llm(self) -> List[str]:
        prompt = self._build_prompt()
        logger.info(f"Calling LLM ({self.model}) for selection. Target: {self.max_features} features.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a credit risk expert specializing in feature engineering and model interpretability."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
            selected = data["selected_features"]
            if "reasoning_summary" in data:
                logger.info(f"LLM Reasoning: {data['reasoning_summary']}")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {content}")
            raise e

        return selected

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if os.path.exists(self.cache_path):
            logger.info(f"Loading cached features from {self.cache_path}")
            with open(self.cache_path, "r") as f:
                self.selected_features = json.load(f)
            return self

        selected = self._call_llm()
        
        # Ensure selected features actually exist in the dataframe
        valid_selected = [f for f in selected if f in X.columns]
        
        if len(valid_selected) == 0:
            raise ValueError("LLM returned 0 features that exist in the dataset.")

        self.selected_features = valid_selected

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.selected_features, f)

        logger.info(f"Successfully selected {len(self.selected_features)} features.")
        return self

    def transform(self, X: pd.DataFrame):
        if self.selected_features is None:
            raise ValueError("Selector must be fitted before transform.")
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        return self.fit(X, y).transform(X)

        