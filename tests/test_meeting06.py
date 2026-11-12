import copy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.meeting06_data import (
    EvidenceError,
    load_evidence,
    stability,
    summarize,
    validate_bcd,
    validate_logs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_counts_and_boundaries():
    e = load_evidence()
    assert len(e["logs"]) == 7880
    assert len(e["qwen"]["per_seed"]) == 30
    assert e["meta"]["bcd_status"] == "COMPLETED_ARCHIVE_AUDITED"
    assert e["meta"]["raw_archive_verified_order_seeds"] == [11]
    values = {(r["split"], r["arm"]): r for r in e["results"]}
    assert values["Validation", "native"]["mean"] == pytest.approx(68.34782608695652)
    assert values["Validation", "native"]["sd"] == pytest.approx(3.2882905045546167)
    assert values["Validation", "model224"]["mean"] == pytest.approx(27.652173913043477)
    assert values["Test", "native"]["mean"] == pytest.approx(76.61538461538461)
    assert all(row["missing"] == 0 for row in e["qwen"]["per_seed"])


@pytest.mark.parametrize("mutation", ["duplicate", "count", "incomplete", "accuracy"])
def test_bad_score_rejected(mutation):
    scores = copy.deepcopy(load_evidence()["qwen"]["per_seed"])
    if mutation == "duplicate":
        scores[0] = copy.deepcopy(scores[6])
    elif mutation == "count":
        scores[0]["correct"] = 999
    elif mutation == "incomplete":
        scores[0]["complete"] = False
    else:
        scores[0]["accuracy"] = 0
    with pytest.raises(EvidenceError):
        summarize(scores)


def test_logs_fail_closed():
    e = load_evidence()
    with pytest.raises(EvidenceError):
        validate_logs(e["logs"][:-1], e["qwen"]["per_seed"])
    bad = copy.deepcopy(e["logs"])
    bad[1] = bad[0]
    with pytest.raises(EvidenceError):
        validate_logs(bad, e["qwen"]["per_seed"])


def test_missing_evidence(tmp_path):
    with pytest.raises(EvidenceError):
        load_evidence(tmp_path / "data/meeting06")


def test_stability_respects_invalids():
    e = load_evidence()
    for split in ["Train", "Validation", "Test"]:
        for arm in ["native", "model224"]:
            s = stability(e["logs"], split, arm)
            assert (
                s["correct_all_five"]
                <= s["stable_valid"]
                <= s["valid_all_five"]
                <= s["total"]
            )
            assert s["invalid_any"] == (
                2 if split == "Train" and arm == "native" else 0
            )


def test_standalone_and_repeated_selections():
    app = AppTest.from_file(str(ROOT / "meeting06_app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "FinIntel AI | Meeting 06"
    assert app.metric[0].value == "68.35%"
    for split in ["Test", "Train", "Validation", "Test", "Validation"]:
        app.selectbox(key="m06_split").set_value(split).run()
        assert not app.exception
        assert len(app.title) == 1
        if split == "Test":
            assert any(
                "Historical Test has already been exposed" in x.value
                for x in app.warning
            )
    for case in [1, 0, 1]:
        app.selectbox(key="m06_case").set_value(case).run()
        assert not app.exception
    assert any("not a final model" in x.value for x in app.warning)


def test_integrated_route_preserves_legacy_navigation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    assert not app.exception
    assert app.title[0].value == "FinIntel AI | Meeting 06"
    options = app.sidebar.radio[0].options
    assert [x.split(" — ")[0] for x in options] == [
        "Meeting 02",
        "Meeting 03",
        "Meeting 04",
        "Meeting 05",
        "Meeting 06",
    ]
    for index in [1, 2, 3, 4, 3, 1, 4]:
        app.sidebar.radio[0].set_value(options[index]).run()
        assert not app.exception
        assert len(app.title) == 1


def test_missing_snapshot_shows_no_fake_metric(monkeypatch, tmp_path):
    import src.meeting06_data as loader

    monkeypatch.setattr(loader, "DATA", tmp_path / "data/meeting06")
    app = AppTest.from_file(str(ROOT / "meeting06_app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 0
    assert any("No substitute results" in x.value for x in app.error)


def test_bcd_verified_results_and_narrative_limit():
    bcd = load_evidence()["bcd"]
    expected = {"original_reference": 56, "B": 41, "C": 57, "D": 62}
    assert bcd["passed"]
    assert bcd["A"] == "NOT_TRAINED"
    for arm, correct in expected.items():
        assert bcd["metrics"][arm]["Validation"]["correct"] == correct
        assert set(bcd["metrics"][arm]) == {"Train", "Validation"}
    assert bcd["paired_validation"]["D_vs_C"] == {
        "fixes": 12,
        "regressions": 7,
        "net_correct": 5,
    }
    assert (
        bcd["metrics"]["C"]["Validation"]["mean_ce"]
        < bcd["metrics"]["D"]["Validation"]["mean_ce"]
    )
    assert bcd["parameter_counts"]["C"] != bcd["parameter_counts"]["D"]
    for split in ["Train", "Validation"]:
        assert (
            bcd["actual_neural_top2_number_presence_diagnostic"][split][
                "printed_number_present"
            ]
            == 0
        )


@pytest.mark.parametrize(
    "mutation", ["unverified", "a", "test", "denominator", "score", "paired", "company"]
)
def test_bcd_bad_evidence_rejected(mutation):
    bcd = copy.deepcopy(load_evidence()["bcd"])
    if mutation == "unverified":
        bcd["passed"] = False
    elif mutation == "a":
        bcd["A"] = "TRAINED"
    elif mutation == "test":
        bcd["metrics"]["D"]["Test"] = {}
    elif mutation == "denominator":
        bcd["metrics"]["D"]["Validation"]["denominator"] = 115
    elif mutation == "score":
        bcd["metrics"]["D"]["Validation"]["exact_cell_accuracy"] = 0.99
    elif mutation == "paired":
        bcd["paired_validation"]["D_vs_C"]["fixes"] = 20
    else:
        bcd["by_company_correct_total"]["CSCO"]["D"][0] = 22
    with pytest.raises(EvidenceError):
        validate_bcd(bcd)


def test_bcd_render_has_results_and_caveats():
    app = AppTest.from_file(str(ROOT / "meeting06_app.py"), default_timeout=30).run()
    assert not app.exception
    assert any(t.label == "B/C/D verified results" for t in app.tabs)
    values = [m.value for m in app.metric]
    for value in ["56/96", "41/96", "57/96", "62/96"]:
        assert value in values
    text = " ".join(
        x.value
        for x in list(app.markdown)
        + list(app.caption)
        + list(app.warning)
        + list(app.info)
    )
    for fragment in [
        "0/537",
        "0/96",
        "without scale conversion",
        "NOT TRAINED",
        "different parameter counts",
        "no with/without-narrative ablation",
    ]:
        assert fragment in text
    assert "No verified B/C/D training outputs" not in text


def test_header_checkpoint_replay_is_separate_from_training():
    e = load_evidence()
    replay, training = e["header_replay"], e["header_training"]
    assert replay["predictions_reproduced"] is True
    assert replay["bitwise_reproduced"] is False
    assert sum(x["denominator"] for x in replay["arms"].values()) == 288
    assert [x["correct"] for x in replay["arms"].values()] == [53, 84, 79]
    assert training["old_training_reproduced"] is False
    assert training["fresh_repeat_all_predictions_byte_identical"] is True
    assert [x["Validation"]["correct"] for x in training["fresh_results"].values()] == [
        56,
        82,
        82,
    ]
    assert [
        x["Validation"]["correct"] for x in training["repeat_results"].values()
    ] == [56, 82, 82]
    assert "earlier" not in e


def test_reproducibility_warnings_render():
    app = AppTest.from_file(str(ROOT / "meeting06_app.py"), default_timeout=30).run()
    assert not app.exception
    text = " ".join(
        x.value
        for x in list(app.markdown)
        + list(app.caption)
        + list(app.warning)
        + list(app.success)
        + list(app.info)
    )
    for fragment in [
        "not an independent neural replay",
        "full training result did not reproduce",
        "H1, H2 and H3 remain unconfirmed",
    ]:
        assert fragment in text
    assert not any(t.label == "Earlier diagnostics" for t in app.tabs)


def test_original_pilot_histories_are_separate_and_interactive():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    for model in ['text_only', 'full_table_text_b', 'baseline', 'adaptive_pilot']:
        at.selectbox(key='m06_original_history').set_value(model).run()
        assert not at.exception
    text = '\n'.join(str(x.value) for x in at.caption)
    assert 'before Qwen internal preprocessing' in text
    assert 'These older runs are separate' in text
