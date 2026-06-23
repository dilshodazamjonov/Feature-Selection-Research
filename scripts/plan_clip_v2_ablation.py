from __future__ import annotations

import argparse
import json


DESCRIPTORS = [
    "missing_rate",
    "unique_ratio",
    "concentration_share",
    "signed_log_mean",
    "log_standard_deviation",
    "clipped_skewness",
    "normalized_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the optional CLIP-v2 leave-one-descriptor-out ablation plan.")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    plan = [{"ablation": "full_v2", "dropped_descriptor": None}]
    plan.extend({"ablation": f"without_{name}", "dropped_descriptor": name} for name in DESCRIPTORS)
    print(
        json.dumps(
            {
                "status": "planned",
                "optional": True,
                "execute": False,
                "oot_used_for_descriptor_selection": False,
                "plans": plan,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
