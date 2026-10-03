"""ARGO — Adaptive Regime-Gated Online learner (AuTrad AI core model)."""
from .features import FEATURES, build_features
from .model import ArgoConfig, ArgoModel, Decision

__all__ = ["FEATURES", "build_features", "ArgoConfig", "ArgoModel", "Decision"]
