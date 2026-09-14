"""
Loads the verified FinIntel AI POC checkpoint files.

These three files are the exact artifacts produced by the source notebook's
own "FINAL END-OF-DAY CHECKPOINT" cell, and were independently re-validated
(11/11 pipeline checks, P@5 = 0.840 recomputed from the stored labels)
before being placed here. Nothing in this module invents, recomputes from
scratch, or substitutes any value -- it only loads what was checkpointed.

If a file is missing or fails to load, callers should surface that clearly
rather than falling back to synthetic data.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

OBJECTS_PATH = DATA_DIR / "finintel_EOD_2026_09_08_objects.pkl"
STATUS_PATH = DATA_DIR / "finintel_EOD_2026_09_08_status.pkl"
TENSORS_PATH = DATA_DIR / "finintel_EOD_2026_09_08_tensors.pt"


@dataclass
class POCData:
    # --- retrieval corpus (15 filings / 5 companies) ---
    item7_corpus: list[dict]
    item7_chunks: list[dict]
    embeddings: np.ndarray  # (515, 384) MiniLM, normalized
    retrieval_eval_df: pd.DataFrame  # 25 rows, manually labeled

    # --- multimodal pipeline (7 clean pairs / 3 companies) ---
    clean_multimodal_df: pd.DataFrame
    visual_suitability_df: pd.DataFrame
    text_tokens: torch.Tensor  # [7, 256, 768]
    text_masks: torch.Tensor  # [7, 256]
    visual_patches: torch.Tensor  # [7, 196, 768]
    text_cross_repr: torch.Tensor  # [7, 768]
    visual_cross_repr: torch.Tensor  # [7, 768]
    cross_attention_fused: torch.Tensor  # [7, 1536]
    adaptive_fused_repr: torch.Tensor  # [7, 768]
    modality_weights: torch.Tensor  # [7, 2]

    # --- POC status summary, as checkpointed ---
    status: dict[str, Any]

    def missing_files(self) -> list[str]:
        return []


class POCDataLoadError(RuntimeError):
    """Raised when one of the three required checkpoint files can't be loaded."""


def _check_files_exist() -> list[str]:
    missing = []
    for p in (OBJECTS_PATH, STATUS_PATH, TENSORS_PATH):
        if not p.exists():
            missing.append(str(p))
    return missing


def load_poc_data() -> POCData:
    missing = _check_files_exist()
    if missing:
        raise POCDataLoadError(
            "Missing required checkpoint file(s):\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nThese must be placed under data/. See README.md."
        )

    try:
        with open(OBJECTS_PATH, "rb") as f:
            objs = pickle.load(f)
        with open(STATUS_PATH, "rb") as f:
            status = pickle.load(f)
        tensors = torch.load(TENSORS_PATH, map_location="cpu", weights_only=False)
    except Exception as e:
        raise POCDataLoadError(f"Failed to load checkpoint files: {e}") from e

    required_obj_keys = [
        "pilot_item7_corpus",
        "pilot_item7_chunks",
        "pilot_retrieval_eval_df",
        "pilot_clean_multimodal_df",
        "pilot_visual_suitability_df",
    ]
    missing_keys = [k for k in required_obj_keys if k not in objs]
    if missing_keys:
        raise POCDataLoadError(
            f"objects.pkl is missing expected key(s): {missing_keys}"
        )

    required_tensor_keys = [
        "pilot_item7_embeddings",
        "clean_text_tokens",
        "clean_text_masks",
        "clean_visual_patches",
        "clean_text_cross_repr",
        "clean_visual_cross_repr",
        "clean_cross_attention_fused",
        "clean_adaptive_fused_repr",
        "clean_modality_weights",
    ]
    missing_tensor_keys = [k for k in required_tensor_keys if k not in tensors]
    if missing_tensor_keys:
        raise POCDataLoadError(
            f"tensors.pt is missing expected key(s): {missing_tensor_keys}"
        )

    embeddings = np.asarray(tensors["pilot_item7_embeddings"], dtype="float32")

    return POCData(
        item7_corpus=objs["pilot_item7_corpus"],
        item7_chunks=objs["pilot_item7_chunks"],
        embeddings=embeddings,
        retrieval_eval_df=objs["pilot_retrieval_eval_df"],
        clean_multimodal_df=objs["pilot_clean_multimodal_df"],
        visual_suitability_df=objs["pilot_visual_suitability_df"],
        text_tokens=tensors["clean_text_tokens"],
        text_masks=tensors["clean_text_masks"],
        visual_patches=tensors["clean_visual_patches"],
        text_cross_repr=tensors["clean_text_cross_repr"],
        visual_cross_repr=tensors["clean_visual_cross_repr"],
        cross_attention_fused=tensors["clean_cross_attention_fused"],
        adaptive_fused_repr=tensors["clean_adaptive_fused_repr"],
        modality_weights=tensors["clean_modality_weights"],
        status=status,
    )


def run_pipeline_validation(data: POCData) -> tuple[dict[str, bool], bool]:
    """
    Re-implements the source notebook's own 11-check "FINAL POC PIPELINE
    VALIDATION" cell against the loaded data, rather than trusting a
    hardcoded '11/11' string. Returns (checks, all_passed).
    """
    checks: dict[str, bool] = {}

    checks["15 MD&A filings available"] = len(data.item7_corpus) == 15
    checks["Retrieval chunks available"] = len(data.item7_chunks) > 0
    checks["Retrieval evaluation available"] = len(data.retrieval_eval_df) > 0
    checks["Real financial pairs >= 5"] = len(data.clean_multimodal_df) >= 5
    checks["Multiple companies >= 3"] = data.clean_multimodal_df["ticker"].nunique() >= 3
    checks["FinBERT tokens valid"] = data.text_tokens.shape[0] == len(data.clean_multimodal_df)
    checks["ViT patches valid"] = data.visual_patches.shape[0] == len(data.clean_multimodal_df)
    checks["Cross-attention fused valid"] = (
        data.cross_attention_fused.shape[0] == len(data.clean_multimodal_df)
    )
    checks["Adaptive fusion valid"] = (
        data.adaptive_fused_repr.shape[0] == len(data.clean_multimodal_df)
    )
    checks["Modality weights valid"] = (
        data.modality_weights.shape[0] == len(data.clean_multimodal_df)
    )

    tensor_list = [
        data.text_tokens,
        data.visual_patches,
        data.cross_attention_fused,
        data.adaptive_fused_repr,
        data.modality_weights,
    ]
    checks["No NaN / Inf"] = all(torch.isfinite(x).all().item() for x in tensor_list)

    return checks, all(checks.values())


def compute_retrieval_metrics(data: POCData) -> dict[str, Any]:
    """
    Recomputes Precision@5 (overall and per-query) directly from the
    manually-labeled pilot_retrieval_eval_df, rather than hardcoding 0.840.
    """
    df = data.retrieval_eval_df
    per_query = df.groupby("query")["relevant"].mean()
    return {
        "queries": per_query.index.tolist(),
        "precision_at_5_by_query": per_query.to_dict(),
        "mean_precision_at_5": float(per_query.mean()),
        "relevant_count": int(df["relevant"].sum()),
        "total_judged": int(len(df)),
    }
