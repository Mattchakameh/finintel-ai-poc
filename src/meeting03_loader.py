"""
Meeting 03 evidence loader for FinIntel AI.

Loads and validates the evidence package exported from the
Meeting 03 research notebook. This module does not recompute,
retrain, or alter reported research results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_EVIDENCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "meeting03_evidence.json"
)


class Meeting03EvidenceError(RuntimeError):
    """Raised when the Meeting 03 evidence package is missing or invalid."""


def load_meeting03_evidence(
    path: str | Path = DEFAULT_EVIDENCE_PATH,
) -> Dict[str, Any]:
    """Load the Meeting 03 evidence JSON without modifying its contents."""

    evidence_path = Path(path)

    if not evidence_path.exists():
        raise Meeting03EvidenceError(
            f"Meeting 03 evidence file not found: {evidence_path}"
        )

    try:
        with evidence_path.open("r", encoding="utf-8") as f:
            evidence = json.load(f)
    except json.JSONDecodeError as exc:
        raise Meeting03EvidenceError(
            f"Meeting 03 evidence JSON is invalid: {exc}"
        ) from exc

    if not isinstance(evidence, dict):
        raise Meeting03EvidenceError(
            "Meeting 03 evidence must be a JSON object."
        )

    return evidence


def validate_meeting03_evidence(
    evidence: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate required sections and internal pilot-count consistency.

    Returns:
        (passed, checks)
    """

    checks: List[str] = []
    passed = True

    required_sections = [
        "metadata",
        "dataset",
        "heldout_results",
        "error_analysis",
        "ablation",
        "metric_separation",
        "statistics_plan",
        "research_boundary",
        "key_findings",
    ]

    for section in required_sections:
        ok = section in evidence
        checks.append(
            f"{'PASS' if ok else 'FAIL'} — required section: {section}"
        )
        passed = passed and ok

    if not passed:
        return False, checks

    dataset = evidence["dataset"]
    results = evidence["heldout_results"]
    ablation = evidence["ablation"]

    consistency_checks = [
        (
            dataset.get("labeled_samples") == 40,
            "pilot dataset contains 40 labeled samples",
        ),
        (
            dataset.get("train_samples") == 34,
            "training partition contains 34 samples",
        ),
        (
            dataset.get("heldout_samples") == 6,
            "temporal held-out partition contains 6 samples",
        ),
        (
            dataset.get("train_samples", 0)
            + dataset.get("heldout_samples", 0)
            == dataset.get("labeled_samples", -1),
            "train + held-out equals total labeled samples",
        ),
        (
            dataset.get("filings") == 5,
            "pilot contains 5 SEC 10-K filings",
        ),
        (
            dataset.get("xbrl_concepts") == 6,
            "pilot contains 6 XBRL concepts",
        ),
        (
            results.get("concept_micro_f1") == 1.0,
            "held-out concept Micro-F1 is 1.000",
        ),
        (
            results.get("value_exact_match") == 0.167,
            "held-out value exact match is 0.167",
        ),
        (
            results.get("joint_extraction") == 0.167,
            "held-out joint extraction is 0.167",
        ),
        (
            len(ablation) == 6,
            "six model-comparison conditions are present",
        ),
    ]

    for ok, label in consistency_checks:
        checks.append(f"{'PASS' if ok else 'FAIL'} — {label}")
        passed = passed and ok

    return passed, checks


def meeting03_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Return dashboard-ready headline metrics."""

    dataset = evidence["dataset"]
    results = evidence["heldout_results"]
    errors = evidence["error_analysis"]
    stats = evidence["statistics_plan"]

    return {
        "samples": dataset["labeled_samples"],
        "filings": dataset["filings"],
        "concepts": dataset["xbrl_concepts"],
        "train_samples": dataset["train_samples"],
        "heldout_samples": dataset["heldout_samples"],
        "concept_micro_f1": results["concept_micro_f1"],
        "value_exact_match": results["value_exact_match"],
        "joint_extraction": results["joint_extraction"],
        "bottleneck": errors["current_bottleneck"],
        "alpha": stats["alpha"],
        "target_power": stats["target_power"],
    }


def load_and_validate_meeting03(
    path: str | Path = DEFAULT_EVIDENCE_PATH,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Convenience function for Streamlit.

    Raises Meeting03EvidenceError if validation fails.
    """

    evidence = load_meeting03_evidence(path)
    valid, checks = validate_meeting03_evidence(evidence)

    if not valid:
        failed = [c for c in checks if c.startswith("FAIL")]
        raise Meeting03EvidenceError(
            "Meeting 03 evidence validation failed: "
            + "; ".join(failed)
        )

    summary = meeting03_summary(evidence)

    return evidence, summary, checks
