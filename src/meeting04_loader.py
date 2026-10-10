from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "meeting04"

class Meeting04EvidenceError(RuntimeError):
    pass

def load_meeting04():
    paths = {
        "summary": DATA / "meeting04_summary.json",
        "predictions": DATA / "predictions_compact.csv",
        "histories": DATA / "histories_compact.csv",
        "candidates": DATA / "meeting04_test_candidates.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise Meeting04EvidenceError("Missing Meeting 04 files: " + ", ".join(missing))

    summary = json.loads(paths["summary"].read_text())
    predictions = pd.read_csv(paths["predictions"])
    histories = pd.read_csv(paths["histories"])
    candidates = pd.read_csv(paths["candidates"])

    checks = {
        "900 financial facts": summary["dataset"]["facts"] == 900,
        "30 companies": summary["dataset"]["companies"] == 30,
        "5 fiscal years": summary["dataset"]["fiscal_years"] == 5,
        "6 financial concepts": summary["dataset"]["metrics"] == 6,
        "20 final runs": predictions[["model","seed"]].drop_duplicates().shape[0] == 20,
        "130 frozen-test targets": predictions["grounding_id"].nunique() == 130,
        "4 candidates per test target": candidates.groupby("grounding_id").size().eq(4).all(),
    }
    if not all(checks.values()):
        raise Meeting04EvidenceError(
            "Validation failed: " + ", ".join(k for k,v in checks.items() if not v)
        )
    return summary, predictions, histories, candidates, checks
