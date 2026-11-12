"""Read-only, hash-checked saved evidence. No model loading or inference."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "meeting06"
SPLITS = {"Train": 543, "Validation": 115, "Test": 130}
ORDERS = [11, 22, 33, 44, 55]
ARMS = ["native", "model224"]


class EvidenceError(ValueError):
    """The supplied saved-output package is missing or inconsistent."""


def summarize(rows):
    results = []
    for split, total in SPLITS.items():
        for arm in ARMS:
            group = [r for r in rows if r["split"] == split and r["arm"] == arm]
            if sorted(r["seed"] for r in group) != ORDERS:
                raise EvidenceError(
                    f"Expected exactly five option orders: {split}/{arm}"
                )
            for row in group:
                if (
                    row["total"] != total
                    or row["observed"] != total
                    or row["missing"] != 0
                    or not row["complete"]
                    or not 0 <= row["correct"] <= total
                ):
                    raise EvidenceError(
                        f"Incomplete or invalid score group: {split}/{arm}"
                    )
                if abs(row["accuracy"] - row["correct"] / total) > 1e-10:
                    raise EvidenceError("Count and accuracy disagree")
            values = [100 * r["correct"] / total for r in group]
            results.append(
                {
                    "split": split,
                    "arm": arm,
                    "n": total,
                    "mean": statistics.mean(values),
                    "sd": statistics.stdev(values),
                }
            )
    if len(rows) != 30:
        raise EvidenceError("Expected 30 score groups")
    return results


def validate_logs(logs, scores):
    if len(logs) != 7880:
        raise EvidenceError("Expected 7,880 saved responses")
    grouped = defaultdict(list)
    seen = set()
    for row in logs:
        key = (row["id"], row["arm"], row["order_seed"])
        if key in seen:
            raise EvidenceError("Duplicate saved response")
        seen.add(key)
        grouped[(row["split"], row["arm"], row["order_seed"])].append(row)
    for score in scores:
        rows = grouped[(score["split"], score["arm"], score["seed"])]
        if (
            len(rows) != score["total"]
            or sum(r["recomputed_correct"] for r in rows) != score["correct"]
        ):
            raise EvidenceError("Saved responses and scores disagree")
    return True


def stability(logs, split, arm):
    grouped = defaultdict(list)
    for row in logs:
        if row["split"] == split and row["arm"] == arm:
            grouped[row["id"]].append(row)
    if len(grouped) != SPLITS[split] or any(
        len(rows) != 5 for rows in grouped.values()
    ):
        raise EvidenceError("Incomplete option-order stability coverage")
    valid = [
        rows for rows in grouped.values() if all(r["status"] == "ok" for r in rows)
    ]
    stable = sum(
        len({r["reconstructed_option_id"] for r in rows}) == 1 for rows in valid
    )
    all_correct = sum(
        all(r["recomputed_correct"] for r in rows) for rows in grouped.values()
    )
    return {
        "total": len(grouped),
        "stable_valid": stable,
        "valid_all_five": len(valid),
        "invalid_any": len(grouped) - len(valid),
        "correct_all_five": all_correct,
    }


def validate_bcd(bcd):
    if bcd["passed"] is not True or bcd["A"] != "NOT_TRAINED":
        raise EvidenceError("B/C/D evidence is not verified or mislabels A")
    if set(bcd["metrics"]) != {"original_reference", "B", "C", "D"}:
        raise EvidenceError("Unexpected B/C/D arms")
    for arm, metrics in bcd["metrics"].items():
        if set(metrics) != {"Train", "Validation"}:
            raise EvidenceError("B/C/D must not include Test results")
        for split, total in [("Train", 537), ("Validation", 96)]:
            r = metrics[split]
            if r["denominator"] != total or not 0 <= r["correct"] <= total:
                raise EvidenceError("Invalid B/C/D denominator or count")
            if abs(r["exact_cell_accuracy"] - r["correct"] / total) > 1e-10:
                raise EvidenceError("B/C/D count and accuracy disagree")
        if bcd["parameter_counts"][arm] <= 0:
            raise EvidenceError("Invalid trainable parameter count")
        company_correct = sum(
            x[arm][0] for x in bcd["by_company_correct_total"].values()
        )
        company_total = sum(x[arm][1] for x in bcd["by_company_correct_total"].values())
        if company_correct != metrics["Validation"]["correct"] or company_total != 96:
            raise EvidenceError("B/C/D company totals disagree")
    for name, pair in bcd["paired_validation"].items():
        left, right = name.split("_vs_")
        net = (
            bcd["metrics"][left]["Validation"]["correct"]
            - bcd["metrics"][right]["Validation"]["correct"]
        )
        if pair["fixes"] - pair["regressions"] != net or pair["net_correct"] != net:
            raise EvidenceError("B/C/D paired counts disagree")
    return True


def load_evidence(data_dir=None):
    directory = Path(data_dir) if data_dir is not None else DATA
    root = directory.parents[1]
    try:
        manifest = json.loads((directory / "file_hashes.json").read_text())
        for relative, expected in manifest.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise EvidenceError("Evidence path escapes the package")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise EvidenceError(f"Evidence hash mismatch: {relative}")
        read = lambda name: json.loads((directory / name).read_text())
        meta, qwen = read("metadata.json"), read("qwen_summary.json")
        if (
            meta["schema_version"] != 2
            or meta["bcd_status"] != "COMPLETED_ARCHIVE_AUDITED"
        ):
            raise EvidenceError("Unsupported evidence snapshot")
        bcd = read("bcd_verified_results.json")
        validate_bcd(bcd)
        results = summarize(qwen["per_seed"])
        logs = [
            json.loads(line)
            for line in (directory / "qwen_saved_responses.jsonl")
            .read_text()
            .splitlines()
        ]
        validate_logs(logs, qwen["per_seed"])
        return {
            "meta": meta,
            "bcd": bcd,
            "qwen": qwen,
            "results": results,
            "logs": logs,
            "header_replay": read("header_checkpoint_replay.json"),
            "header_training": read("header_training_comparison.json"),
            "examples": read("examples.json"),
        }
    except EvidenceError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EvidenceError(
            f"Saved evidence unavailable or invalid: {error}"
        ) from error
