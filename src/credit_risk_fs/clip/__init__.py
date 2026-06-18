"""CLIP-readiness helpers for feature-selection research.

This package intentionally contains artifact validation and dry-run manifest
helpers only. It does not train CLIP models.
"""

from credit_risk_fs.clip.schemas import (
    ClipDatasetRole,
    ClipDatasetSpec,
    ClipEvidenceAudit,
    ClipFieldRole,
    ClipFieldSpec,
    ClipSourceArtifact,
    ClipTrainingManifest,
)

__all__ = [
    "ClipDatasetRole",
    "ClipDatasetSpec",
    "ClipEvidenceAudit",
    "ClipFieldRole",
    "ClipFieldSpec",
    "ClipSourceArtifact",
    "ClipTrainingManifest",
]
