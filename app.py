"""
FinIntel AI -- Multimodal Financial Intelligence POC
Streamlit presentation layer over the existing, already-validated POC.

This file is a UI/demo layer only. It does not retrain, re-derive, or
alter any of the POC's reported results -- it loads the checkpointed
outputs of the source notebook and displays them. See README.md for
provenance of every file under data/ and assets/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.meeting03_loader import (
    Meeting03EvidenceError,
    load_and_validate_meeting03,
)
from src.meeting04_view import render_meeting04
from src.meeting05_view import render_meeting05
from src.meeting06_view import render_meeting06

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="FinIntel AI -- POC Demo",
    layout="wide",
)
# ============================================================
# Meeting navigation
# ============================================================


MEETING07_EVIDENCE = [(1,
  'Verify earlier results',
  '24 orders · 105/115 for each method · 0 decisions changed',
  'Saved scores only. No inference on shuffled inputs and no protected test run.',
  [('Original replay.py · all option orders and synthetic tie check',
    'import itertools\n'
    '\n'
    'ORDERS=tuple(itertools.permutations(range(4)))\n'
    '\n'
    'def top(scores, order, abstain):\n'
    '    high=max(scores[i] for i in order)\n'
    '    winners=[i for i in order if scores[i]==high]\n'
    '    return None if abstain and len(winners)>1 else winners[0]\n'
    '\n'
    "if __name__=='__main__':\n"
    '    assert top([0.9,0.1,0.9,0.2],(0,1,2,3),False)==0\n'
    '    assert top([0.9,0.1,0.9,0.2],(2,1,0,3),False)==2\n'
    '    assert all(top([0.9,0.1,0.9,0.2],x,True) is None for x in ORDERS)\n'
    '    assert all(top([0.1,0.3,0.9,0.2],x,True)==2 for x in ORDERS)',
    'Temporal selector: 105/115 correct across 24 orders; 0 decisions changed.\n'
    'Locked fusion: 105/115 correct across 24 orders; 0 decisions changed.\n'
    'Synthetic tie: abstained in all 24 orders.')]),
 (2,
  'Explain the extraction pilot',
  'Model C 57/96 · Model D 62/96 on the same Validation questions',
  'Fixed epoch 5. Five more correct cells for D; the paired difference was not statistically '
  'significant. The cell displays archived scores; it does not rerun the models.',
  [('Original Colab · cell 3: Archived Validation results',
    '# Display archived results only. Do not rerun the models.\n'
    'print("Request 2 | Archived Validation results | fixed epoch 5")\n'
    'for model in ("C", "D"):\n'
    '    scores = results[model]["archived_fixed_epoch_5_metrics"]["Validation"]\n'
    '    correct, total = scores["correct"], scores["denominator"]\n'
    '    print(f"Model {model}: {correct}/{total} correct ({correct / total:.2%})")\n'
    'print("Source: validation_curves.json | No model rerun in this notebook.")',
    'Request 2 | Archived Validation results | fixed epoch 5\n'
    'Model C: 57/96 correct (59.38%)\n'
    'Model D: 62/96 correct (64.58%)\n'
    'Source: validation_curves.json | No model rerun in this notebook.')]),
 (3,
  'Check sharper table images',
  'Archived 224 px New8: C 6/32 · D 4/32',
  'The original 224 px notebook recounts saved decisions. The separately reported 336 px run had '
  'unchanged counts; its exact runner was not recovered by this notebook.',
  [('Original Colab · cell 2: exact-cell scoring excerpt',
    'issuers = sorted({issuer for issuer, _, _ in rows})\n'
    'assert len(rows) == 32 and len(issuers) == 8\n'
    'print("New8 | Archived 224 px frozen-checkpoint results")\n'
    'print("Companies:", ", ".join(issuers))\n'
    'print("Questions:", len(rows))\n'
    'for i, model in enumerate(("C", "D"), start=1):\n'
    '    print(f"Model {model}: {sum(row[i] for row in rows)}/{len(rows)} exact cells")\n'
    'print("This cell recounts saved decisions; it does not run either model.")',
    'New8 | Archived 224 px frozen-checkpoint results\n'
    'Companies: ABBV, AOS, DE, DOV, PH, SWK, UPS, V\n'
    'Questions: 32\n'
    'Model C: 6/32 exact cells\n'
    'Model D: 4/32 exact cells\n'
    'This cell recounts saved decisions; it does not run either model.')]),
 (4,
  'Evaluate the balanced challenge',
  '32 base questions × 6 variants · standalone rule: 150/192 correct',
  'The four-choice rule answered 162/192. C and D took the separate 32-question exact-cell test, not '
  'this four-choice test.',
  [('Original Colab · cell 2: balance and scoring excerpt',
    'from collections import Counter\n'
    '\n'
    '# Frozen labels and predictions, joined by example ID after prediction freeze.\n'
    'variants_per_question = Counter(base for base, _, _, _ in rows)\n'
    'positions = Counter("ABCD"[gold] for _, _, gold, _ in rows)\n'
    'categories = Counter(kind for _, kind, _, _ in rows)\n'
    'assert len(variants_per_question) == 32 and set(variants_per_question.values()) == {6}\n'
    'assert positions == dict.fromkeys("ABCD", 48)\n'
    'assert len(categories) == 6 and set(categories.values()) == {32}\n'
    'correct = sum(gold == selected for _, _, gold, selected in rows)\n'
    'answered = sum(selected is not None for _, _, _, selected in rows)\n'
    'assert (correct, answered, len(rows)) == (150, 162, 192)\n'
    'print(f"{len(variants_per_question)} base questions × 6 distractor types = {len(rows)} variants")\n'
    'print("Correct positions:", dict(sorted(positions.items())))\n'
    'print(f"Standalone rule: {correct}/{len(rows)} correct; {answered}/{len(rows)} answered.")\n'
    'print("C and D did not take this four-choice test.")',
    '32 base questions × 6 distractor types = 192 variants\n'
    "Correct positions: {'A': 48, 'B': 48, 'C': 48, 'D': 48}\n"
    'Standalone rule: 150/192 correct; 162/192 answered.\n'
    'C and D did not take this four-choice test.')]),
 (5,
  'Plan the independent evaluation',
  'Planning target: 144 companies × 24 questions = 3,456',
  'This estimates sample size under stated assumptions, not accuracy. The independent test has not run.',
  [('Original Colab · sample-size calculation (complete cell)',
    'from math import ceil, sqrt\n'
    'from statistics import NormalDist\n'
    '\n'
    'FAMILY_ALPHA = 0.05\n'
    'TEST_COUNT = 3\n'
    'TARGET_POWER = 0.80\n'
    'MINIMUM_GAIN = 0.05\n'
    'QUESTIONS_PER_ISSUER = 24\n'
    'TARGET_ISSUERS = 144\n'
    'normal = NormalDist()\n'
    'critical = normal.inv_cdf(1 - FAMILY_ALPHA / (2 * TEST_COUNT))\n'
    'z_power = normal.inv_cdf(TARGET_POWER)\n'
    '\n'
    'def required_issuers(paired_discordance, issuer_icc):\n'
    '    variance = paired_discordance - MINIMUM_GAIN ** 2\n'
    '    design_effect = 1 + (QUESTIONS_PER_ISSUER - 1) * issuer_icc\n'
    '    questions = (critical + z_power)**2 * variance * design_effect / MINIMUM_GAIN**2\n'
    '    return ceil(questions / QUESTIONS_PER_ISSUER)\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    pilot_q = (12 + 7) / 96\n'
    '    print(f"Historical extraction discordance (planning proxy only): {pilot_q:.4f}")\n'
    '    for q, rho in [(0.25, 0.05), (0.25, 0.10), (0.25, 0.20), (0.35, 0.10)]:\n'
    '        print(f"q={q:.2f}, ICC={rho:.2f}: {required_issuers(q, rho)} issuers")\n'
    '    assert required_issuers(0.25, 0.10) == 143\n'
    '    print(f"Proposed target: {TARGET_ISSUERS} issuers × {QUESTIONS_PER_ISSUER} = "\n'
    '          f"{TARGET_ISSUERS * QUESTIONS_PER_ISSUER} questions.")',
    'Historical extraction discordance (planning proxy only): 0.1979\n'
    'q=0.25, ICC=0.05: 93 issuers\n'
    'q=0.25, ICC=0.10: 143 issuers\n'
    'q=0.25, ICC=0.20: 242 issuers\n'
    'q=0.35, ICC=0.10: 201 issuers\n'
    'Proposed target: 144 issuers × 24 = 3456 questions.')]),
 (6,
  'Check input coverage and replay',
  '6A: 0 tokens dropped across 2,632 inputs · 6B: 192/192 saved choices matched',
  'The archived replay and the corrected-input coverage check are separate. This is not a new training '
  'run.',
  [('Original Colab · cell 17: 6A input-coverage checks',
    'source_archive = INPUTS / "FinIntel_sourcefix5_v1_Data_and_Proof.zip"\n'
    'print(run("test_gate.py", source_archive).strip())\n'
    'coverage = json.loads(run("window_audit.py", source_archive, INPUTS / "tokenizer"))\n'
    'print(json.dumps(coverage, indent=2))\n'
    'assert coverage["windows_per_option"] == {\n'
    '    "Train": {"1": 2156, "3": 16}, "Validation": {"1": 456, "3": 4}\n'
    '}\n'
    'assert coverage["source_tokens_dropped"] == 0\n'
    'assert coverage["unequal_window_count_within_target"] == 0\n'
    'assert coverage["model_ready"] is False',
    'Verified local CPU output (not a Colab execution):\n'
    'PASS: immutable SHA-pinned input; byte tamper rejected; model admission denied with no model '
    'payload\n'
    '{\n'
    '  "windows_per_option": {\n'
    '    "Train": {\n'
    '      "1": 2156,\n'
    '      "3": 16\n'
    '    },\n'
    '    "Validation": {\n'
    '      "1": 456,\n'
    '      "3": 4\n'
    '    }\n'
    '  },\n'
    '  "unequal_window_count_within_target": 0,\n'
    '  "source_tokens_dropped": 0,\n'
    '  "model_ready": false\n'
    '}'),
   ('Original Colab · cell 20: 6B two-run replay excerpt',
    '    with TemporaryDirectory(prefix="finintel-r6-") as folder:\n'
    '        outputs = []\n'
    '        for label in ("first", "second"):\n'
    '            destination = Path(folder) / label\n'
    '            run("replay_archived.py", *args, "--out", destination)\n'
    '            outputs.append({\n'
    '                file.name: file.read_bytes()\n'
    '                for file in (destination / "fixed_checkpoint_inference").glob("*.json")\n'
    '            })\n'
    '        assert len(outputs[0]) == 7 and outputs[0] == outputs[1], "Replay files differ"\n'
    '        summary = json.loads(outputs[0]["summary.json"])\n'
    '\n'
    '    for model in ("C", "D"):\n'
    '        score = summary["metrics"]["archived_global"][model]\n'
    '        check = summary["comparisons"][model]\n'
    '        assert check["historical_cell_decisions_agree"] == score["denominator"] == 96\n'
    '        print(f"{model}: {score[\'correct\']}/96 correct; 96/96 archived selections matched; "\n'
    '              f"maximum historical logit delta = '
    '{check[\'historical_max_abs_logit_delta\']:.9g}")\n'
    '    print("Two deterministic reruns: all seven replay JSON files are byte-identical.")\n'
    '    Path("request6_historical_replay_summary.json").write_text(json.dumps(summary, indent=2))',
    'Verified local CPU output (not a Colab execution):\n'
    'Verified prior CPU replay: C 57/96, D 62/96; 192/192 archived choices matched.\n'
    'This is a summary of the prior run, not a new result.')])]
MEETING07_OCR = ('print("OCR read accuracy (480x128, found crops only):")\nfor split in ["train", "validation", "test"]:\n    s = ocr["ocr_accuracy_per_split"][split]["480"]\n    print(f"  {split:<11} header exact {s[\'header_exact_acc\']:.0%} "\n          f"({s[\'header_exact\']}/{s[\'n_found\']}), "\n          f"year printed {s[\'year_printed_acc\']:.0%}, "\n          f"row prefix-12 {s[\'row_prefix12_acc\']:.0%}")', 'OCR read accuracy (480x128, found crops only):\n  train       header exact 100% (3823/3823), year printed 100%, row prefix-12 100%\n  validation  header exact 100% (1351/1351), year printed 100%, row prefix-12 100%\n  test        header exact 100% (1302/1302), year printed 100%, row prefix-12 100%')
MEETING07_PILOT_ARCHIVE = [('Step 19 · Master comparison: value-only to OCR headers (historical)', 'Same 102 original-panel questions. The old OCR experiment changed the inputs and later audit identified an answer-key-derived date. The apparent gain is not an OCR-only effect or a valid headline result.', '    ("Original pilot, value-only features", "A", int(round(pilot["model_A"]["exact_cell"] * 102))),\n    ("Original pilot, value-only features", "B", int(round(pilot["model_B"]["exact_cell"] * 102))),\n    ("Original pilot, value-only features", "C", int(round(pilot["model_C"]["exact_cell"] * 102))),\n    ("Original pilot, value-only features", "D", int(round(pilot["model_D"]["exact_cell"] * 102))),\n    ("OCR headers (RapidOCR, no training)", "B", ocr["heads"]["B_ocr"]["test"]["correct"]),\n    ("OCR headers (RapidOCR, no training)", "C", ocr["heads"]["C_ocr"]["test"]["correct"]),\n    ("OCR headers (RapidOCR, no training)", "D", ocr["heads"]["D_ocr"]["test"]["correct"]),', '  Original pilot, value-only features    model A:   8/102 (7.84%)\n  Original pilot, value-only features    model B:  37/102 (36.27%)\n  Original pilot, value-only features    model C:  29/102 (28.43%)\n  Original pilot, value-only features    model D:  31/102 (30.39%)\n  OCR headers (RapidOCR, no training)    model B:  86/102 (84.31%)\n  OCR headers (RapidOCR, no training)    model C:  98/102 (96.08%)\n  OCR headers (RapidOCR, no training)    model D:  74/102 (72.55%)'), ('Step 20 · OCR + ISO, three-seed C/D (historical; date leakage)', 'Original panel, 102 questions per seed. The exact fiscal year-end feature used the answer key; these are not valid test results.', 'three = json.loads((PILOT_DIR / "three_seed_c_d_results.json").read_text())\n# Saved output for the three-seed check is shown below.', 'Three seeds, OCR headers + ISO, test (102 questions, full-tuple):\n  Seed       C     D     D-C\n  11        99    96   -2.94 pp\n  22        95    97   +1.96 pp\n  33        99    95   -3.92 pp\n  mean   97.67 96.00   -1.63 pp'), ('Step 23 · Nine Model D gate variants (historical; date leakage)', 'The provided Colab cell shows similarity and FinBERT variants, not ConvLSTM. These scores use the affected date feature and are excluded from performance comparisons.', 'hdr = f"{\'Variant\':<26}{\'Val/103\':>8}{\'Orig D\':>16}{\'Orig D-C pp\':>14}{\'Fresh D\':>16}{\'Fresh D-C pp\':>14}"\nprint(hdr)\nfor name, v in gate["variants"].items():\n    o = v["summary_original"]["all"]\n    f = v["summary_fresh"]["all"]\n    print(f"{name:<26}{v[\'val_mean_over_103\']:>8.2f}"\n          f"{str(o[\'D_per_seed\']):>16}{o[\'mean_diff_pp\']:>+13.2f} pp"\n          f"{str(f[\'D_per_seed\']):>16}{f[\'mean_diff_pp\']:>+13.2f} pp")\nprint()', 'Saved-output header: Gate improvement: 9 variants, seeds 11/22/33, OCR headers + ISO, test n = 102 per panel, full-tuple. C reference is fixed (not retrained).\nAffected accuracy rows withheld.'), ('Step 25 · Frozen ViT vs. ViT + OCR-similarity gate (historical diagnostic)', 'Each panel has 102 questions per seed. Only D gets the OCR similarity inputs; this does not isolate the learned gate or the image contribution. The date provenance of this old result was not certified for the corrected analysis.', 's = vit["summary_D_vit_shrink0.5_vs_C_vit"]\nprint("ViT pixels + OCR-similarity gate (D_sim_shrink0.5), test n = 102:")\nfor panel in ["original", "fresh"]:\n    a = s[panel]["all"]\n    print(f"  {panel:<8} C_vit {a[\'C_per_seed\']} (mean {a[\'C_mean_sd\'][\'mean\']:.2f}), "\n          f"D_vit {a[\'D_per_seed\']} (mean {a[\'D_mean_sd\'][\'mean\']:.2f}), "\n          f"mean D-C {a[\'mean_diff_pp\']:+.2f} pp, 95% CI "\n          f"{a[\'mean_diff_bootstrap_ci95_pp\']} pp")\n    ao = s[panel]["answerable_only"]\n    print(f"           answerable-only: mean D-C {ao[\'mean_diff_pp\']:+.2f} pp, "', 'ViT pixels + OCR-similarity gate (D_sim_shrink0.5), test n = 102:\n  original C_vit [31, 31, 37] (mean 33.00), D_vit [67, 68, 78] (mean 71.00), mean D-C +37.25 pp, 95% CI [26.14, 48.05] pp\n  fresh    C_vit [39, 38, 40] (mean 39.00), D_vit [74, 74, 81] (mean 76.33), mean D-C +36.60 pp, 95% CI [25.49, 47.39] pp'), ('Step 26 · Cross-checking stored results', '62/62 checks matched the stored reports. This does not rerun inference or certify whether an earlier model’s input contained leaked features.', 'def check(label, got, want, tol=0.0005):\n    if isinstance(want, list):\n        ok = (len(got) == len(want)\n              and all(abs(g - w) <= tol for g, w in zip(got, want)))\n    else:\n        ok = (got == want) if tol == 0 else abs(got - want) <= tol\n    checks.append((label, got, want, ok))', 'OK   model_A original pilot                 got 0.0784313725490196 want 0.0784313725490196\n  OK   model_B original pilot                 got 0.3627450980392157 want 0.3627450980392157\n  OK   model_C original pilot                 got 0.28431372549019607 want 0.28431372549019607\n  OK   model_D original pilot                 got 0.30392156862745096 want 0.30392156862745096\n62 checks, 62 passed.')]
MEETING07_NOTEBOOK = '{\n "nbformat": 4,\n "nbformat_minor": 4,\n "metadata": {\n  "kernelspec": {\n   "display_name": "Python 3",\n   "language": "python",\n   "name": "python3"\n  }\n },\n "cells": [\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "# Meeting 7 evidence excerpts\\n",\n    "\\n",\n    "Read-only evidence. Most code excerpts need the private source inputs and are not standalone. The Request 5 power calculation is a complete runnable cell."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 1: Verify earlier results\\n",\n    "\\n",\n    "24 orders \\u00b7 105/115 for each method \\u00b7 0 decisions changed\\n",\n    "\\n",\n    "Saved scores only. No inference on shuffled inputs and no protected test run."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original replay.py \\u00b7 all option orders and synthetic tie check\\n",\n    "\\n",\n    "```python\\n",\n    "import itertools\\n",\n    "\\n",\n    "ORDERS=tuple(itertools.permutations(range(4)))\\n",\n    "\\n",\n    "def top(scores, order, abstain):\\n",\n    "    high=max(scores[i] for i in order)\\n",\n    "    winners=[i for i in order if scores[i]==high]\\n",\n    "    return None if abstain and len(winners)>1 else winners[0]\\n",\n    "\\n",\n    "if __name__==\'__main__\':\\n",\n    "    assert top([0.9,0.1,0.9,0.2],(0,1,2,3),False)==0\\n",\n    "    assert top([0.9,0.1,0.9,0.2],(2,1,0,3),False)==2\\n",\n    "    assert all(top([0.9,0.1,0.9,0.2],x,True) is None for x in ORDERS)\\n",\n    "    assert all(top([0.1,0.3,0.9,0.2],x,True)==2 for x in ORDERS)\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "Temporal selector: 105/115 correct across 24 orders; 0 decisions changed.\\n",\n    "Locked fusion: 105/115 correct across 24 orders; 0 decisions changed.\\n",\n    "Synthetic tie: abstained in all 24 orders.\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 2: Explain the extraction pilot\\n",\n    "\\n",\n    "Model C 57/96 \\u00b7 Model D 62/96 on the same Validation questions\\n",\n    "\\n",\n    "Fixed epoch 5. Five more correct cells for D; the paired difference was not statistically significant. The cell displays archived scores; it does not rerun the models."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 cell 3: Archived Validation results\\n",\n    "\\n",\n    "```python\\n",\n    "# Display archived results only. Do not rerun the models.\\n",\n    "print(\\"Request 2 | Archived Validation results | fixed epoch 5\\")\\n",\n    "for model in (\\"C\\", \\"D\\"):\\n",\n    "    scores = results[model][\\"archived_fixed_epoch_5_metrics\\"][\\"Validation\\"]\\n",\n    "    correct, total = scores[\\"correct\\"], scores[\\"denominator\\"]\\n",\n    "    print(f\\"Model {model}: {correct}/{total} correct ({correct / total:.2%})\\")\\n",\n    "print(\\"Source: validation_curves.json | No model rerun in this notebook.\\")\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "Request 2 | Archived Validation results | fixed epoch 5\\n",\n    "Model C: 57/96 correct (59.38%)\\n",\n    "Model D: 62/96 correct (64.58%)\\n",\n    "Source: validation_curves.json | No model rerun in this notebook.\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 3: Check sharper table images\\n",\n    "\\n",\n    "Archived 224 px New8: C 6/32 \\u00b7 D 4/32\\n",\n    "\\n",\n    "The original 224 px notebook recounts saved decisions. The separately reported 336 px run had unchanged counts; its exact runner was not recovered by this notebook."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 cell 2: exact-cell scoring excerpt\\n",\n    "\\n",\n    "```python\\n",\n    "issuers = sorted({issuer for issuer, _, _ in rows})\\n",\n    "assert len(rows) == 32 and len(issuers) == 8\\n",\n    "print(\\"New8 | Archived 224 px frozen-checkpoint results\\")\\n",\n    "print(\\"Companies:\\", \\", \\".join(issuers))\\n",\n    "print(\\"Questions:\\", len(rows))\\n",\n    "for i, model in enumerate((\\"C\\", \\"D\\"), start=1):\\n",\n    "    print(f\\"Model {model}: {sum(row[i] for row in rows)}/{len(rows)} exact cells\\")\\n",\n    "print(\\"This cell recounts saved decisions; it does not run either model.\\")\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "New8 | Archived 224 px frozen-checkpoint results\\n",\n    "Companies: ABBV, AOS, DE, DOV, PH, SWK, UPS, V\\n",\n    "Questions: 32\\n",\n    "Model C: 6/32 exact cells\\n",\n    "Model D: 4/32 exact cells\\n",\n    "This cell recounts saved decisions; it does not run either model.\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 4: Evaluate the balanced challenge\\n",\n    "\\n",\n    "32 base questions \\u00d7 6 variants \\u00b7 standalone rule: 150/192 correct\\n",\n    "\\n",\n    "The four-choice rule answered 162/192. C and D took the separate 32-question exact-cell test, not this four-choice test."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 cell 2: balance and scoring excerpt\\n",\n    "\\n",\n    "```python\\n",\n    "from collections import Counter\\n",\n    "\\n",\n    "# Frozen labels and predictions, joined by example ID after prediction freeze.\\n",\n    "variants_per_question = Counter(base for base, _, _, _ in rows)\\n",\n    "positions = Counter(\\"ABCD\\"[gold] for _, _, gold, _ in rows)\\n",\n    "categories = Counter(kind for _, kind, _, _ in rows)\\n",\n    "assert len(variants_per_question) == 32 and set(variants_per_question.values()) == {6}\\n",\n    "assert positions == dict.fromkeys(\\"ABCD\\", 48)\\n",\n    "assert len(categories) == 6 and set(categories.values()) == {32}\\n",\n    "correct = sum(gold == selected for _, _, gold, selected in rows)\\n",\n    "answered = sum(selected is not None for _, _, _, selected in rows)\\n",\n    "assert (correct, answered, len(rows)) == (150, 162, 192)\\n",\n    "print(f\\"{len(variants_per_question)} base questions \\u00d7 6 distractor types = {len(rows)} variants\\")\\n",\n    "print(\\"Correct positions:\\", dict(sorted(positions.items())))\\n",\n    "print(f\\"Standalone rule: {correct}/{len(rows)} correct; {answered}/{len(rows)} answered.\\")\\n",\n    "print(\\"C and D did not take this four-choice test.\\")\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "32 base questions \\u00d7 6 distractor types = 192 variants\\n",\n    "Correct positions: {\'A\': 48, \'B\': 48, \'C\': 48, \'D\': 48}\\n",\n    "Standalone rule: 150/192 correct; 162/192 answered.\\n",\n    "C and D did not take this four-choice test.\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 5: Plan the independent evaluation\\n",\n    "\\n",\n    "Planning target: 144 companies \\u00d7 24 questions = 3,456\\n",\n    "\\n",\n    "This estimates sample size under stated assumptions, not accuracy. The independent test has not run."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 sample-size calculation (complete cell)\\n",\n    "\\n",\n    "```python\\n",\n    "from math import ceil, sqrt\\n",\n    "from statistics import NormalDist\\n",\n    "\\n",\n    "FAMILY_ALPHA = 0.05\\n",\n    "TEST_COUNT = 3\\n",\n    "TARGET_POWER = 0.80\\n",\n    "MINIMUM_GAIN = 0.05\\n",\n    "QUESTIONS_PER_ISSUER = 24\\n",\n    "TARGET_ISSUERS = 144\\n",\n    "normal = NormalDist()\\n",\n    "critical = normal.inv_cdf(1 - FAMILY_ALPHA / (2 * TEST_COUNT))\\n",\n    "z_power = normal.inv_cdf(TARGET_POWER)\\n",\n    "\\n",\n    "def required_issuers(paired_discordance, issuer_icc):\\n",\n    "    variance = paired_discordance - MINIMUM_GAIN ** 2\\n",\n    "    design_effect = 1 + (QUESTIONS_PER_ISSUER - 1) * issuer_icc\\n",\n    "    questions = (critical + z_power)**2 * variance * design_effect / MINIMUM_GAIN**2\\n",\n    "    return ceil(questions / QUESTIONS_PER_ISSUER)\\n",\n    "\\n",\n    "if __name__ == \\"__main__\\":\\n",\n    "    pilot_q = (12 + 7) / 96\\n",\n    "    print(f\\"Historical extraction discordance (planning proxy only): {pilot_q:.4f}\\")\\n",\n    "    for q, rho in [(0.25, 0.05), (0.25, 0.10), (0.25, 0.20), (0.35, 0.10)]:\\n",\n    "        print(f\\"q={q:.2f}, ICC={rho:.2f}: {required_issuers(q, rho)} issuers\\")\\n",\n    "    assert required_issuers(0.25, 0.10) == 143\\n",\n    "    print(f\\"Proposed target: {TARGET_ISSUERS} issuers \\u00d7 {QUESTIONS_PER_ISSUER} = \\"\\n",\n    "          f\\"{TARGET_ISSUERS * QUESTIONS_PER_ISSUER} questions.\\")\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "Historical extraction discordance (planning proxy only): 0.1979\\n",\n    "q=0.25, ICC=0.05: 93 issuers\\n",\n    "q=0.25, ICC=0.10: 143 issuers\\n",\n    "q=0.25, ICC=0.20: 242 issuers\\n",\n    "q=0.35, ICC=0.10: 201 issuers\\n",\n    "Proposed target: 144 issuers \\u00d7 24 = 3456 questions.\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "code",\n   "execution_count": null,\n   "metadata": {},\n   "outputs": [],\n   "source": [\n    "from math import ceil, sqrt\\n",\n    "from statistics import NormalDist\\n",\n    "\\n",\n    "FAMILY_ALPHA = 0.05\\n",\n    "TEST_COUNT = 3\\n",\n    "TARGET_POWER = 0.80\\n",\n    "MINIMUM_GAIN = 0.05\\n",\n    "QUESTIONS_PER_ISSUER = 24\\n",\n    "TARGET_ISSUERS = 144\\n",\n    "normal = NormalDist()\\n",\n    "critical = normal.inv_cdf(1 - FAMILY_ALPHA / (2 * TEST_COUNT))\\n",\n    "z_power = normal.inv_cdf(TARGET_POWER)\\n",\n    "\\n",\n    "def required_issuers(paired_discordance, issuer_icc):\\n",\n    "    variance = paired_discordance - MINIMUM_GAIN ** 2\\n",\n    "    design_effect = 1 + (QUESTIONS_PER_ISSUER - 1) * issuer_icc\\n",\n    "    questions = (critical + z_power)**2 * variance * design_effect / MINIMUM_GAIN**2\\n",\n    "    return ceil(questions / QUESTIONS_PER_ISSUER)\\n",\n    "\\n",\n    "if __name__ == \\"__main__\\":\\n",\n    "    pilot_q = (12 + 7) / 96\\n",\n    "    print(f\\"Historical extraction discordance (planning proxy only): {pilot_q:.4f}\\")\\n",\n    "    for q, rho in [(0.25, 0.05), (0.25, 0.10), (0.25, 0.20), (0.35, 0.10)]:\\n",\n    "        print(f\\"q={q:.2f}, ICC={rho:.2f}: {required_issuers(q, rho)} issuers\\")\\n",\n    "    assert required_issuers(0.25, 0.10) == 143\\n",\n    "    print(f\\"Proposed target: {TARGET_ISSUERS} issuers \\u00d7 {QUESTIONS_PER_ISSUER} = \\"\\n",\n    "          f\\"{TARGET_ISSUERS * QUESTIONS_PER_ISSUER} questions.\\")"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## Request 6: Check input coverage and replay\\n",\n    "\\n",\n    "6A: 0 tokens dropped across 2,632 inputs \\u00b7 6B: 192/192 saved choices matched\\n",\n    "\\n",\n    "The archived replay and the corrected-input coverage check are separate. This is not a new training run."\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 cell 17: 6A input-coverage checks\\n",\n    "\\n",\n    "```python\\n",\n    "source_archive = INPUTS / \\"FinIntel_sourcefix5_v1_Data_and_Proof.zip\\"\\n",\n    "print(run(\\"test_gate.py\\", source_archive).strip())\\n",\n    "coverage = json.loads(run(\\"window_audit.py\\", source_archive, INPUTS / \\"tokenizer\\"))\\n",\n    "print(json.dumps(coverage, indent=2))\\n",\n    "assert coverage[\\"windows_per_option\\"] == {\\n",\n    "    \\"Train\\": {\\"1\\": 2156, \\"3\\": 16}, \\"Validation\\": {\\"1\\": 456, \\"3\\": 4}\\n",\n    "}\\n",\n    "assert coverage[\\"source_tokens_dropped\\"] == 0\\n",\n    "assert coverage[\\"unequal_window_count_within_target\\"] == 0\\n",\n    "assert coverage[\\"model_ready\\"] is False\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "Verified local CPU output (not a Colab execution):\\n",\n    "PASS: immutable SHA-pinned input; byte tamper rejected; model admission denied with no model payload\\n",\n    "{\\n",\n    "  \\"windows_per_option\\": {\\n",\n    "    \\"Train\\": {\\n",\n    "      \\"1\\": 2156,\\n",\n    "      \\"3\\": 16\\n",\n    "    },\\n",\n    "    \\"Validation\\": {\\n",\n    "      \\"1\\": 456,\\n",\n    "      \\"3\\": 4\\n",\n    "    }\\n",\n    "  },\\n",\n    "  \\"unequal_window_count_within_target\\": 0,\\n",\n    "  \\"source_tokens_dropped\\": 0,\\n",\n    "  \\"model_ready\\": false\\n",\n    "}\\n",\n    "```"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "### Original Colab \\u00b7 cell 20: 6B two-run replay excerpt\\n",\n    "\\n",\n    "```python\\n",\n    "    with TemporaryDirectory(prefix=\\"finintel-r6-\\") as folder:\\n",\n    "        outputs = []\\n",\n    "        for label in (\\"first\\", \\"second\\"):\\n",\n    "            destination = Path(folder) / label\\n",\n    "            run(\\"replay_archived.py\\", *args, \\"--out\\", destination)\\n",\n    "            outputs.append({\\n",\n    "                file.name: file.read_bytes()\\n",\n    "                for file in (destination / \\"fixed_checkpoint_inference\\").glob(\\"*.json\\")\\n",\n    "            })\\n",\n    "        assert len(outputs[0]) == 7 and outputs[0] == outputs[1], \\"Replay files differ\\"\\n",\n    "        summary = json.loads(outputs[0][\\"summary.json\\"])\\n",\n    "\\n",\n    "    for model in (\\"C\\", \\"D\\"):\\n",\n    "        score = summary[\\"metrics\\"][\\"archived_global\\"][model]\\n",\n    "        check = summary[\\"comparisons\\"][model]\\n",\n    "        assert check[\\"historical_cell_decisions_agree\\"] == score[\\"denominator\\"] == 96\\n",\n    "        print(f\\"{model}: {score[\'correct\']}/96 correct; 96/96 archived selections matched; \\"\\n",\n    "              f\\"maximum historical logit delta = {check[\'historical_max_abs_logit_delta\']:.9g}\\")\\n",\n    "    print(\\"Two deterministic reruns: all seven replay JSON files are byte-identical.\\")\\n",\n    "    Path(\\"request6_historical_replay_summary.json\\").write_text(json.dumps(summary, indent=2))\\n",\n    "```\\n",\n    "\\n",\n    "Saved or previously verified output:\\n",\n    "\\n",\n    "```text\\n",\n    "Verified local CPU output (not a Colab execution):\\n",\n    "Verified prior CPU replay: C 57/96, D 62/96; 192/192 archived choices matched.\\n",\n    "This is a summary of the prior run, not a new result.\\n",\n    "```"\n   ]\n  }\n ]\n}'

MEETING07_STAGES = [(1,
  'Set up the Python environment',
  'Install/check NumPy, pandas, Matplotlib and SciPy; document separate OCR and encoder requirements.',
  'The saved run used NumPy 2.4.6, pandas 2.2.3 and SciPy 1.16.3. The pandas and SciPy pin attempts did not '
  'succeed.',
  'Setup, not model evidence.',
  [{'Cell': 1,
    'Cell title': 'FinIntel AI — SEC Pilot: Models A/B/C/D in Google Colab (v4)',
    'Notebook record': 'Explanation',
    'Step': 1,
    'Type': 'markdown'},
   {'Cell': 2,
    'Cell title': 'Step 1 — Install pinned libraries',
    'Notebook record': 'Explanation',
    'Step': 1,
    'Type': 'markdown'},
   {'Cell': 3,
    'Cell title': 'Step 1 - install the pinned libraries this notebook needs.',
    'Notebook record': 'Saved output',
    'Step': 1,
    'Type': 'code'}]),
 (2,
  'Connect to the source files',
  'Mount Drive and identify the project folder.',
  'The saved notebook found its project inputs. The setup cells do not produce model results.',
  'Private paths are omitted.',
  [{'Cell': 4,
    'Cell title': 'Step 2 — Mount Google Drive and find the pilot folder',
    'Notebook record': 'Explanation',
    'Step': 2,
    'Type': 'markdown'},
   {'Cell': 5,
    'Cell title': 'Step 2a - connect Google Drive (Colab only).',
    'Notebook record': 'Saved output',
    'Step': 2,
    'Type': 'code'},
   {'Cell': 6,
    'Cell title': 'Step 2b - find the pilot folder.',
    'Notebook record': 'Saved output',
    'Step': 2,
    'Type': 'code'}]),
 (3,
  'Load SEC 10-K / XBRL data',
  'Read the raw and cleaned grids for 36 companies, fiscal 2022–2024, and five concepts: revenue, net '
  'income, assets, liabilities and equity.',
  '540 possible facts, 465 found raw and 480 after cleaning.',
  'These are source counts, not test accuracy.',
  [{'Cell': 7,
    'Cell title': 'Step 3 — Load the SEC data',
    'Notebook record': 'Explanation',
    'Step': 3,
    'Type': 'markdown'},
   {'Cell': 8,
    'Cell title': 'Step 3 - load the raw SEC data and the cleaned table, plus settings.',
    'Notebook record': 'Saved output',
    'Step': 3,
    'Type': 'code'}]),
 (4,
  'Inspect the raw data before cleaning',
  'Chart missingness by concept and split; inspect scale and negative values.',
  '75 of 540 attempted facts were missing, including 34 liabilities facts. The saved audit reviewed 11 '
  'negative values as losses or negative equity.',
  'No missing value was filled by the diagnostic.',
  [{'Cell': 9,
    'Cell title': 'Step 4 — Visual EDA of the RAW data, before cleaning',
    'Notebook record': 'Explanation',
    'Step': 4,
    'Type': 'markdown'},
   {'Cell': 10,
    'Cell title': 'Step 4 - visual EDA of the RAW data, before any cleaning.',
    'Notebook record': 'Saved output',
    'Step': 4,
    'Type': 'code'}]),
 (5,
  'Clean and audit the SEC data',
  'Fix issuer lookup and audit annual versus quarterly figures, tags, units and outliers.',
  '15 missing facts recovered, 15 existing values corrected, 60 still missing, 0 rows deleted. The 21 '
  'flagged extreme values were retained after review.',
  'Two questions in the original test had a known printed-value/XBRL mismatch.',
  [{'Cell': 11,
    'Cell title': 'Step 5 — Cleaning',
    'Notebook record': 'Explanation',
    'Step': 5,
    'Type': 'markdown'},
   {'Cell': 12,
    'Cell title': 'Step 5 - cleaning: compare raw vs cleaned, and check the result.',
    'Notebook record': 'Saved output',
    'Step': 5,
    'Type': 'code'}]),
 (6,
  'Encode the questions and candidate cells',
  'In the baseline, question-side features encode the requested concept and year. Candidate features '
  'describe values, not the candidate’s concept or year. The example has 15 candidates.',
  'The encoding demonstration is not a score.',
  'The presence of a gold label in the explanatory example is not permission to use it as an input.',
  [{'Cell': 13,
    'Cell title': 'Step 6 — Feature engineering',
    'Notebook record': 'Explanation',
    'Step': 6,
    'Type': 'markdown'},
   {'Cell': 14,
    'Cell title': 'Step 6 - feature engineering, worked through on one real question.',
    'Notebook record': 'Saved output',
    'Step': 6,
    'Type': 'code'}]),
 (7,
  'Split by company and audit the first leakage issue',
  'Separate issuers before scoring. Remove candidate features that explicitly revealed the encoded target '
  'and probe the remaining features one at a time.',
  '20/8/8 companies; 275 Train, 103 Validation, 102 original test questions. One-feature checks ranged from '
  '17 to 36 correct out of 480.',
  'That early fix is separate from the later answer-key date issue in the header/OCR pipeline. The '
  'contaminated initial scores are excluded.',
  [{'Cell': 15,
    'Cell title': 'Step 7 — Company-grouped split and leakage checks',
    'Notebook record': 'Explanation',
    'Step': 7,
    'Type': 'markdown'},
   {'Cell': 16,
    'Cell title': 'Step 7 - check the split, build the questions, encode them.',
    'Notebook record': 'Saved output',
    'Step': 7,
    'Type': 'code'},
   {'Cell': 17,
    'Cell title': 'Step 7 (continued) — Leakage check (read this before trusting any number)',
    'Notebook record': 'Explanation',
    'Step': 7,
    'Type': 'markdown'},
   {'Cell': 18,
    'Cell title': 'Step 7 - leakage check. Replays the two checks from leakage_check.py',
    'Notebook record': 'Saved output',
    'Step': 7,
    'Type': 'code'}]),
 (8,
  'Measure evidence retrieval',
  'Check whether the correct candidate is among the top k before evaluating which cell the model selects.',
  'On the original test, recall was 21.57% at k=1, 57.84% at k=3, 72.55% at k=5 and 95.10% at k=10.',
  'This is candidate recall, not extraction accuracy.',
  [{'Cell': 19,
    'Cell title': 'Step 8 — Retrieval top-k (recall)',
    'Notebook record': 'Explanation',
    'Step': 8,
    'Type': 'markdown'},
   {'Cell': 20,
    'Cell title': 'Step 8 - retrieval top-k recall, measured separately from selection.',
    'Notebook record': 'Saved output',
    'Step': 8,
    'Type': 'code'}]),
 (9,
  'Train the four value-only baselines',
  'A uses question features; B adds table/value features; C uses equal fixed weighting; D learns a gate. '
  'Checkpoints are chosen on Validation.',
  'Saved checkpoint Validation: A 12.62%, B 43.69%, C 33.98%, D 42.72%. D’s selected checkpoint was epoch '
  '20.',
  'The optional 160-run grid cell has no saved output; these baselines do not use OCR or table pixels.',
  [{'Cell': 21,
    'Cell title': 'Step 9 — Features and models A–D',
    'Notebook record': 'Explanation',
    'Step': 9,
    'Type': 'markdown'},
   {'Cell': 22,
    'Cell title': 'Step 9 - features and models A-D. The verified pilot code is reused,',
    'Notebook record': 'Saved output',
    'Step': 9,
    'Type': 'code'},
   {'Cell': 23,
    'Cell title': 'Step 9 — Train A, B, C and D',
    'Notebook record': 'Explanation',
    'Step': 9,
    'Type': 'markdown'},
   {'Cell': 24,
    'Cell title': 'Step 9 - train A, B, C and D with the best configs from best_config.json.',
    'Notebook record': 'Saved output',
    'Step': 9,
    'Type': 'code'},
   {'Cell': 25,
    'Cell title': 'OPTIONAL - re-run the full tuning grid (160 runs, about a minute on CPU).',
    'Notebook record': 'No saved output',
    'Step': 9,
    'Type': 'code'}]),
 (10,
  'Evaluate exact extraction and compare with the archive',
  'Require the amount, unit, row and fiscal period to match on the original 102 questions. Compare the saved '
  'run, current notebook rerun and restricted top-k results.',
  'B 37/102, C 29/102 and D 31/102 matched. A was 8/102 in the archive and 15/102 in the notebook rerun, '
  'explicitly flagged as a mismatch. D’s 54 year-only mistakes were 54 of its 71 errors.',
  'A’s two runs are not interchangeable. The notebook’s rule comparator was not a learned model.',
  [{'Cell': 26,
    'Cell title': 'Step 10 — Evaluate once on test',
    'Notebook record': 'Explanation',
    'Step': 10,
    'Type': 'markdown'},
   {'Cell': 27,
    'Cell title': 'Step 10 - evaluate once on the 102 test questions.',
    'Notebook record': 'Saved output',
    'Step': 10,
    'Type': 'code'},
   {'Cell': 28,
    'Cell title': 'Check: do the numbers above match the saved pilot results?',
    'Notebook record': 'Saved output',
    'Step': 10,
    'Type': 'code'},
   {'Cell': 29,
    'Cell title': 'Step 10 (continued) - accuracy when the model may only pick among top-k.',
    'Notebook record': 'Saved output',
    'Step': 10,
    'Type': 'code'}]),
 (11,
  'Check overfitting and early stopping',
  'Use Validation-based early stopping (patience 5) and restore the best checkpoint. Inspect training and '
  'Validation curves separately.',
  'For value-only D, Train rose from 34.91% to 36.73% between epochs 20 and 25, while Validation fell from '
  '42.72% to 37.86%. The run restored epoch 20.',
  'This describes one archived training curve, not a claim that overfitting was eliminated.',
  [{'Cell': 30,
    'Cell title': 'Step 11 — Overfitting',
    'Notebook record': 'Explanation',
    'Step': 11,
    'Type': 'markdown'},
   {'Cell': 31,
    'Cell title': 'Step 11 - overfitting: controls, per-epoch curves, train/val/test gap.',
    'Notebook record': 'Saved output',
    'Step': 11,
    'Type': 'code'}]),
 (12,
  'Run the simulated-header diagnostic',
  'Supply simulated header text to see whether the scoring pipeline can use explicit matching fields.',
  'Excluded from real performance results.',
  'Synthetic headers are not evidence read from a real filing; simulated scores are not displayed.',
  [{'Cell': 32,
    'Cell title': 'Step 12 — Optional diagnostic: header text (SIMULATED ONLY — DO NOT QUOTE)',
    'Notebook record': 'Explanation',
    'Step': 12,
    'Type': 'markdown'},
   {'Cell': 33,
    'Cell title': 'Step 12 - OPTIONAL DIAGNOSTIC. Read the warning in the markdown above',
    'Notebook record': 'Saved output',
    'Step': 12,
    'Type': 'code'}]),
 (13,
  'Try text extracted from real 10-K headers',
  'Replace simulated headers with row and fiscal-period strings from the source tables.',
  'The old original-panel report listed B 83/102, C 91/102 and D 65/102.',
  'Historical only: the later audit identified an answer-key-derived question date even in the old non-ISO '
  'header path. The comparison is not clean.',
  [{'Cell': 34,
    'Cell title': 'Step 13 — Real 10-K headers (text, extracted — not simulated)',
    'Notebook record': 'Explanation',
    'Step': 13,
    'Type': 'markdown'},
   {'Cell': 35,
    'Cell title': 'Step 13 - real extracted 10-K headers. Loads the saved run; nothing',
    'Notebook record': 'Saved output',
    'Step': 13,
    'Type': 'code'}]),
 (14,
  'Try pixels with random feature encoders',
  'Compare a frozen random CNN and a raw-pixel projection; use shuffled crops as a control.',
  'Archived C on the original 102 questions: 28 with the random CNN versus 30 shuffled; 25 with the '
  'raw-pixel projection versus 26 shuffled.',
  'Substitution diagnostics, not tests of adding pixels to an unchanged text baseline.',
  [{'Cell': 36,
    'Cell title': 'Step 14 — Real table pixels, random encoders (negative result)',
    'Notebook record': 'Explanation',
    'Step': 14,
    'Type': 'markdown'},
   {'Cell': 37,
    'Cell title': 'Step 14 - real pixels with random encoders (stored results by default).',
    'Notebook record': 'Saved output',
    'Step': 14,
    'Type': 'code'}]),
 (15,
  'Use frozen ViT and FinBERT',
  'Encode real table crops with pretrained ViT features and the text side with FinBERT; probe fiscal-period '
  'separability.',
  'Archived original panel: B 36/102, C 34/102, D 35/102. The full ViT period probe scored 36.86% versus '
  'approximately 33% chance.',
  'The encoder change prevents a pixel-only causal comparison; this is a diagnostic, not fine-tuning.',
  [{'Cell': 38,
    'Cell title': 'Step 15 — Real ViT + FinBERT (frozen, pretrained)',
    'Notebook record': 'Explanation',
    'Step': 15,
    'Type': 'markdown'},
   {'Cell': 39,
    'Cell title': 'Step 15 - real ViT + FinBERT (stored results by default).',
    'Notebook record': 'Saved output',
    'Step': 15,
    'Type': 'code'}]),
 (16,
  'Read printed table headers with RapidOCR',
  'Use local RapidOCR on 480×128 crop strips; separately evaluate extraction with header text inputs.',
  'Header transcription: 3,823/3,823 Train, 1,351/1,351 Validation and 1,302/1,302 original-test crops '
  'found.',
  'These are crop-transcription denominators, not questions. The old extraction-score rows also used the '
  'later-flagged answer-key date.',
  [{'Cell': 40,
    'Cell title': 'Step 16 — OCR: the visual information was there all along',
    'Notebook record': 'Explanation',
    'Step': 16,
    'Type': 'markdown'},
   {'Cell': 41,
    'Cell title': 'Step 16 - OCR read accuracy and heads with OCR headers.',
    'Notebook record': 'Saved output',
    'Step': 16,
    'Type': 'code'}]),
 (17,
  'Record the scope of a separate model smoke check',
  'The notebook documents a small model smoke check and a hardware limitation for a larger run.',
  'No large-model result from this notebook is used in the Meeting 7 comparison.',
  'The smoke check cannot establish performance and is outside the displayed model comparison.',
  [{'Cell': 42,
    'Cell title': 'Step 17 — Qwen-VL: what ran, and what could not run here',
    'Notebook record': 'Explanation',
    'Step': 17,
    'Type': 'markdown'},
   {'Cell': 43,
    'Cell title': 'Step 17 - Qwen-VL sample (stored only). No download is started here.',
    'Notebook record': 'Saved output',
    'Step': 17,
    'Type': 'code'}]),
 (18,
  'Try 15 D variants with OCR and dates',
  'Compare weight decay, hard negatives, table dropout and alternative gates on Validation and the original '
  'test.',
  'The archived comparison did not produce an adjusted significant gain. Some gates saturated Validation '
  'without transferring to test.',
  'These old OCR + ISO runs use the affected date feature. Do not treat the high scores as clean or use them '
  'to select a final model.',
  [{'Cell': 44,
    'Cell title': 'Step 18 — Model D improvement with OCR headers (15 variants)',
    'Notebook record': 'Explanation',
    'Step': 18,
    'Type': 'markdown'},
   {'Cell': 45,
    'Cell title': 'Step 18 - Model D improvement with OCR headers (stored results).',
    'Notebook record': 'Saved output',
    'Step': 18,
    'Type': 'code'}]),
 (19,
  'Assemble the original 102-question master table',
  'Put historical value-only, real-header, pixel and OCR outputs side by side without treating them as one '
  'controlled ablation.',
  'Archived C is 29/102 in the value-only row and 98/102 in the old OCR-header row.',
  'The later date audit invalidates that apparent gain as evidence for OCR alone. The selected original '
  'output appears under code below.',
  [{'Cell': 46,
    'Cell title': 'Step 19 — Master comparison table and hypothesis verdicts',
    'Notebook record': 'Explanation',
    'Step': 19,
    'Type': 'markdown'},
   {'Cell': 47,
    'Cell title': 'Step 19 - master comparison table (test, n = 102, full-tuple).',
    'Notebook record': 'Saved output',
    'Step': 19,
    'Type': 'code'}]),
 (20,
  'Repeat C and D over three seeds',
  'Use the same original 102 questions per seed (11, 22, 33) and choose configurations on Validation only.',
  'Historical OCR + ISO means: C 97.67 and D 96.00 out of 102; D−C −1.63 points. None of the seed '
  'comparisons was significant.',
  'Answer-key date issue: these are archived numbers, not valid clean test performance.',
  [{'Cell': 48,
    'Cell title': 'Step 20 — Three-seed C vs D (original panel)',
    'Notebook record': 'Explanation',
    'Step': 20,
    'Type': 'markdown'},
   {'Cell': 49,
    'Cell title': 'Step 20 - three-seed C vs D, original 102-question panel (stored results).',
    'Notebook record': 'Saved output',
    'Step': 20,
    'Type': 'code'}]),
 (21,
  'Check OCR and text on new issuers',
  'Use 102 questions from eight companies outside the original 36 and retain the original Train and '
  'Validation splits. Flag three unanswerable questions before scoring.',
  'Use the separate corrected OCR rerun for the current comparison: C 95.00 and D 97.67 of 102 on average. '
  'It is not the archived Step 21 output.',
  'The old OCR + ISO and extracted-text runs used the affected date pipeline. The corrected rerun is listed '
  'separately in the pilot overview.',
  [{'Cell': 50,
    'Cell title': 'Step 21 — Fresh-panel test, part 1: OCR text and the text control',
    'Notebook record': 'Explanation',
    'Step': 21,
    'Type': 'markdown'},
   {'Cell': 51,
    'Cell title': 'Step 21 - fresh panel, part 1: OCR text + ISO, three seeds, and the',
    'Notebook record': 'Saved output',
    'Step': 21,
    'Type': 'code'}]),
 (22,
  'Check frozen pixels on the fresh panel',
  'Compare frozen ViT representations to shuffled crops and examine fiscal-year errors; the fresh panel '
  'contains 102 questions.',
  'Archived three-seed means: C 39.00 and D 40.33 of 102. D−C +1.31 points, with interval [−2.94, +5.56].',
  'No convincing gain; the original design did not run a true missing/degraded-visual condition. It did not '
  'isolate H1 image addition.',
  [{'Cell': 52,
    'Cell title': 'Step 22 — Fresh-panel test, part 2: real ViT pixels and H3',
    'Notebook record': 'Explanation',
    'Step': 22,
    'Type': 'markdown'},
   {'Cell': 53,
    'Cell title': 'Step 22 - fresh panel, part 2: real ViT pixels and H3 (stored results).',
    'Notebook record': 'Saved output',
    'Step': 22,
    'Type': 'code'}]),
 (23,
  'Compare nine improved gate variants',
  'Test shrunk similarities, a linear gate, regularization and FinBERT-based similarity or scoring; use '
  'seeds 11, 22 and 33.',
  'The supplied cell is a readout of gate variants in the OCR + ISO condition, not a ConvLSTM run. It did '
  'not demonstrate a reliable D advantage.',
  'The affected date feature excludes these saved high scores from clean comparisons.',
  [{'Cell': 54,
    'Cell title': 'Step 23 — Model D gate improvement: 9 new gate variants, both panels',
    'Notebook record': 'Explanation',
    'Step': 23,
    'Type': 'markdown'},
   {'Cell': 55,
    'Cell title': 'Step 23 - Model D gate improvement, text condition (stored results).',
    'Notebook record': 'Saved output',
    'Step': 23,
    'Type': 'code'}]),
 (24,
  'Diagnose the FinBERT scorer and linear gate',
  'Compare full- and projected-embedding similarities; inspect a linear gate’s Validation-to-test behavior.',
  'Saved row-similarity gap: 0.062 in full FinBERT and 0.063 after projection. Year gap fell from 0.056 to '
  '0.028, so projection did not preserve every signal.',
  'A diagnostic of this feature pipeline, not a general claim about FinBERT. Do not use affected accuracy '
  'rows as clean results.',
  [{'Cell': 56,
    'Cell title': 'Step 24 — Two diagnostics: why FinBERT collapses, and why the linear gate hurts',
    'Notebook record': 'Explanation',
    'Step': 24,
    'Type': 'markdown'},
   {'Cell': 57,
    'Cell title': 'Step 24 - the two diagnostics, recomputed from the stored results.',
    'Notebook record': 'Saved output',
    'Step': 24,
    'Type': 'code'}]),
 (25,
  'Compare frozen ViT with a gate that also sees OCR',
  'C uses frozen ViT; D additionally receives OCR similarity features. A prior version that read audit '
  'strings was discarded.',
  'Historical per-102 means: original C 33.00, D 71.00; fresh C 39.00, D 76.33, each across three seeds.',
  'Inputs were unmatched; this does not prove an image or learned-fusion gain. This historical output is not '
  'certified under the later date audit.',
  [{'Cell': 58,
    'Cell title': 'Step 25 — The ViT stage: an OCR rescue, not a fusion win',
    'Notebook record': 'Explanation',
    'Step': 25,
    'Type': 'markdown'},
   {'Cell': 59,
    'Cell title': 'Step 25 - ViT stage: shrunk similarity gate on real ViT pixels (stored results).',
    'Notebook record': 'Saved output',
    'Step': 25,
    'Type': 'code'}]),
 (26,
  'Verify the stored figures against their artifacts',
  'Check the saved result files, including earlier model stages and gate runs; do not run inference.',
  '62 written checks matched their stored artifacts. The baseline rows matched A/B/C/D = 8/37/29/31 out of '
  '102.',
  'Reproducing a stored number does not make its input design valid. The separate Step 10 Model A rerun '
  'mismatch remains.',
  [{'Cell': 60,
    'Cell title': 'Step 26 — Final verification: does this notebook reproduce the reports?',
    'Notebook record': 'Explanation',
    'Step': 26,
    'Type': 'markdown'},
   {'Cell': 61,
    'Cell title': 'Step 26 - final verification of every key number used in this notebook.',
    'Notebook record': 'Saved output',
    'Step': 26,
    'Type': 'code'}]),
 (27,
  'Summarize the notebook and carry forward open work',
  'Separate data verification, retrieval, the original model baselines, OCR transcription and newer model '
  'diagnostics.',
  'No new run in the summary cell. Corrected findings and historical outputs are kept separate in this '
  'review.',
  'Next step: PCA evaluation is in progress.',
  [{'Cell': 62,
    'Cell title': 'Step 27 — Summary and what is still running',
    'Notebook record': 'Explanation',
    'Step': 27,
    'Type': 'markdown'}])]
MEETING07_STAGE_EXCERPTS = {3: ('print("Raw rows (company x year x concept):", raw_data.shape)\n'
     'print("Raw found:", (raw_data["status"] == "found").sum(), "of", len(raw_data),\n'
     'print("Cleaned found:", (data["status"] == "found").sum(), "of", len(data),\n'
     'print()\n'
     'print("Companies: 36 | Years:", config["years"], "| Concepts:", config["concepts"])\n'
     'print("Seeds (split / model init / data order / other):",\n'
     'print()',
     'Raw rows (company x year x concept): (540, 16)\n'
     'Raw found: 465 of 540 | raw missing: 75\n'
     'Cleaned found: 480 of 540 | cleaned missing: 60'),
 5: ('print(f"Missing values: raw {int(raw_missing.sum())} -> cleaned {int(clean_missing.sum())}")\n'
     'print(f"  (15 facts were newly found, 60 confirmed missing, 0 rows deleted)")\n'
     'print(f"Values corrected (same fact, different number): {int(changed.sum())}")\n'
     'print("  Examples: JNJ 2023 Revenues 21,395,000,000 -> 85,159,000,000")\n'
     'print("            (quarterly figure replaced by the annual total);")\n'
     'print("            BRK-B 2022 Revenues 207,771,000,000 -> 302,089,000,000")\n'
     'print("            (total revenue line wins over the contract tag).")\n'
     'print(f"Newly found facts: {len(newly_found)} (all 15 are Exxon - wrong CIK fixed:")\n'
     'print("  first fetch used CIK 2115436, real company is CIK 34088)")\n'
     'print(f"\\nOutlier check (IQR fence on log10 values, cleaned data): "\n'
     'print("  They were inspected one by one: all are genuine large-company")\n'
     'print("  values (Apple / Amazon revenues, Berkshire assets), not errors,")\n'
     'print("  so none was removed or capped.")\n'
     'print("\\nKnown cases kept as they are (not patched):")\n'
     'print("  - PepsiCo 2023: printed number differs from the XBRL tag by $383M")\n'
     'print("    (2 facts) - those 2 test questions are effectively unanswerable.")\n'
     'print("  - McDonald\'s scale fix (fresh panel): \'shares in Thousands\' is")\n'
     'print("    not the dollar scale - see Step 5 text and real_header_extract.py.")\n'
     'print()\n'
     'print("Cleaning log (first lines of cleaning_log.md):")\n'
     'print("\\n".join(log_lines[:12]))\n'
     'print("  ... full record in cleaning_log.md (every row, old and new value).")',
     'Missing values: raw 75 -> cleaned 60\n'
     '  (15 facts were newly found, 60 confirmed missing, 0 rows deleted)\n'
     'Values corrected (same fact, different number): 15'),
 7: ('print("Split problems:",\n'
     'print("\\nQuestions per split (480 total):")\n'
     'print(questions.groupby("split").size().to_string())\n'
     'print("\\nFirst questions:")\n'
     'print(questions.head(3).to_string(index=False))\n'
     'print("\\nEncoded questions per split:",',
     'Split problems: none - 20 train, 8 validation, 8 test, no company in two splits\n'
     '\n'
     'Questions per split (480 total):\n'
     'split\n'
     'test          102\n'
     'train         275\n'
     'validation    103'),
 8: ('print("Recall@k on test - is the gold chunk among the top-k?")\n'
     'for k in [1, 3, 5, 10, 15]:\n'
     'print("\\nReading: recall@k caps accuracy - a model can never score above")\n'
     'print("the recall at the k it is allowed to use. B, C and D need the full")\n'
     'print("candidate set and plateau around k=10, so retrieval is the")\n'
     'print("bottleneck at small k and selection is the bottleneck at large k.")\n'
     'print("Accuracy restricted to top-k is measured in Step 10, after training.")',
     'Recall@k on test - is the gold chunk among the top-k?\n'
     '  k=1 : 21.57%\n'
     '  k=3 : 57.84%\n'
     '  k=5 : 72.55%\n'
     '  k=10: 95.10%\n'
     '  k=15: 100.00%'),
 10: ('for name in ["model_A", "model_B", "model_C", "model_D"]:\n'
      '    print(f"  {name}: this run {fresh:.4f} vs saved {saved_value:.4f} "',
      'Comparison with the saved test_results.json:\n'
      '  model_A: this run 0.1471 vs saved 0.0784 -> MISMATCH\n'
      '  model_B: this run 0.3627 vs saved 0.3627 -> match\n'
      '  model_C: this run 0.2843 vs saved 0.2843 -> match\n'
      '  model_D: this run 0.3039 vs saved 0.3039 -> match'),
 11: ('print("Controls actually applied:")\n'
      'print("  - early stopping on validation exact-cell accuracy, patience 5,")\n'
      'print("    best checkpoint restored (epochs never picked on test)")\n'
      'print("  - weight decay, dropout on the fused vector, AdamW, cosine")\n'
      'print("    learning-rate decay to 0, gradient clipping at norm 1.0")\n'
      'print("  - company-grouped split (20/8/8), no tuning on test")\n'
      'print()\n'
      'print("Train / validation / test gap per model:")\n'
      'print(f"{\'Model\':<8}{\'Train\':>10}{\'Validation\':>12}{\'Test\':>10}{\'Gap\':>8}")\n'
      'for name in ["A", "B", "C", "D"]:\n'
      '    print(f"{name:<8}{train_acc:>10.2%}{val_acc:>12.2%}{test_acc:>10.2%}"\n'
      'print("\\nThe gaps are small: with the fixed features the models cannot")\n'
      'print("memorise their way to the answer - there is nothing in the")\n'
      'print("features worth memorising.")\n'
      'print()\n'
      'print("Per-epoch curves - where validation peaked (model D):")\n'
      'for row in histories["D"]:\n'
      '    print(f"  epoch {row[\'epoch\']:>2}: train {row[\'train_exact_cell\']:.2%} "\n'
      'print("\\nD overfitted after epoch 20: training kept rising (34.91% to")\n'
      'print("36.73%) while validation fell (42.72% to 37.86%). Early stopping")\n'
      'print("cut the run there and restored the epoch-20 checkpoint.")\n'
      'print("\\nValidation accuracy vs training size (from learning_curves.csv):")\n'
      'print(size_curve.pivot(index="model", columns="train_fraction",\n'
      'for ax, name in zip(axes.flat, ["A", "B", "C", "D"]):',
      '  epoch 20: train 34.91% validation 42.72%\n'
      '  epoch 25: train 36.73% validation 37.86%\n'
      'D overfitted after epoch 20: training kept rising (34.91% to'),
 14: ('for label, fname in [("v1 frozen random CNN", "real_pixel_results.json"),\n'
      '    print(f"Pixel {label}, test (102 questions, full-tuple):")\n'
      '    print()\n'
      'print("Text baselines for comparison (real headers, Step 15):")\n'
      'print("  B 83/102, C 91/102, D 65/102")\n'
      'print("Reading: the pixel channel added nothing over shuffled noise")\n'
      'print("(C_pixel vs C_shuffled: -1.96 pp, p = 0.6875 in v1; -0.98 pp, p = 1.0")\n'
      'print("in v2). Period accuracy is 30-40% (chance 33%): the year lives only")\n'
      'print("in the header glyphs, and a random encoder cannot read them.")\n'
      'print()\n'
      'print("One sample crop (Q0001, Revenues 2022) is shown below.")\n'
      '    print("Sample crop not found:", sample)\n'
      'print("To re-run from scratch:")\n'
      'print("  python3 real_pixel_render.py")\n'
      'print("  ~/finbert_venv/bin/python real_pixel_encode.py")\n'
      'print("  python3 r',
      'Pixel v1 frozen random CNN, test (102 questions, full-tuple):\n'
      '  B_pixel   33/102 (32.35%)  row 86.27%  period 37.25%\n'
      '  C_pixel   28/102 (27.45%)  row 86.27%  period 33.33%\n'
      '  D_pixel   32/102 (31.37%)  row 79.41%  period 40.20%\n'
      '  C_shuffled (control): 30/102 (29.41%)\n'
      'Pixel v2 raw-pixel projection, test (102 questions, full-tuple):\n'
      '  B_pixel   39/102 (38.24%)  row 95.10%  period 40.20%\n'
      '  C_pixel   25/102 (24.51%)  row 87.25%  period 30.39%\n'
      '  D_pixel   25/102 (24.51%)  row 77.45%  period 35.29%\n'
      '  C_shuffled (control): 26/102 (25.49%)'),
 15: ('print("Real ViT + FinBERT, test (102 questions, full-tuple):")\n'
      'for name in ["B_vit", "C_vit", "D_vit"]:\n'
      '    print(f"  {name:<8} {r[\'test_correct\']}/102 ({r[\'test_exact_cell\']:.2%})"\n'
      'print(f"  C_shuffled (control): {vit[\'C_shuffled\'][\'test_correct\']}/102")\n'
      'print()\n'
      'print("Paired statistics (102 test questions):")\n'
      'print("  C_vit vs C_shuffled:",\n'
      'print()\n'
      'print("Encoder diagnostics (no training):")\n'
      'print(f"  ViT projected cosine, within-question: "\n'
      'print(f"  ViT centroid probe (full 768-dim CLS, train centroids, test crops):")\n'
      'print(f"    period {diag[\'vit_centroid_probe_period_test\']:.2%} (chance ~33%),"\n'
      'print(f"  FinBERT cosine: gold row {fb[\'gold_row_cosine_mean\']:.3f} vs other rows "\n'
      'print()\n'
      'print("with a real image encoder. ViT reads the row la',
      'Real ViT + FinBERT, test (102 questions, full-tuple):\n'
      '  B_vit    36/102 (35.29%)  row 91.18%  period 38.24%\n'
      '  C_vit    34/102 (33.33%)  row 90.20%  period 36.27%\n'
      '  D_vit    35/102 (34.31%)  row 89.22%  period 37.25%\n'
      '  C_shuffled (control): 25/102'),
 22: ('print("Stage 3 - real ViT pixels, seed 22, test (102 questions, full-tuple):")\n'
      'print(f"  B_vit {fv[\'B_vit\']}, C_vit {fv[\'C_vit\']}, D_vit {fv[\'D_vit\']}, "\n'
      'print("  Wrong-year errors dominate every ViT variant, as on the original")\n'
      'print("  panel: period accuracy 38-45% (chance 33%), row accuracy 95-96%.")\n'
      'print()\n'
      'print("Three-seed ViT pixels:")\n'
      'print(f"  C {fvt[\'C_per_seed\']} (mean {fvt[\'C_mean\']:.2f}), "\n'
      'print(f"  McNemar per seed p = {fvt[\'mcnemar_per_seed_D_vs_C_p\']} "\n'
      'print()\n'
      'print("H1 (fresh panel): C_vit vs matched B_text:")\n'
      'print(f"  C_vit {h1[\'C_vit\']} vs B_text {h1[\'B_text\']}: "\n'
      'print(f"  Issuer-level 95% CI (bootstrap over the 8 issuers): "\n'
      'print("  Per issuer (C_vit - B_text):", en',
      'Three-seed ViT pixels:\n'
      '  C [39, 38, 40] (mean 39.00), D [36, 46, 39] (mean 40.33), mean D-C +1.31 pp, 95% CI [-2.94, '
      '5.56] pp\n'
      '  McNemar per seed p = [0.6291, 0.1849, 1.0] (none significant). H2 on pixels: NOT supported.'),
 24: ('print("FinBERT cosine separation, original test, OCR strings (gold vs other):")\n'
      'print(f"  Full 768-dim: row {d[\'full768_gold_row\']:.3f} vs "\n'
      'print(f"  Projected 8-dim: row {d[\'proj8_gold_row\']:.3f} vs "\n'
      'print("  Hash encoder, same test (from the report, section 5): row gap 0.600,")\n'
      'print("  year gap 0.396 - about 10x the FinBERT gap. The projection preserves")\n'
      'print("  the small FinBERT gap, so the projection is NOT the culprit.")\n'
      'print()\n'
      'print("FinBERT scorer, test means (102 questions):")\n'
      'for name in ["C_finbert"',
      'FinBERT cosine separation, original test, OCR strings (gold vs other):\n'
      '  Full 768-dim: row 0.801 vs 0.739 (gap 0.062); year 0.994 vs 0.938 (gap 0.056)\n'
      '  Projected 8-dim: row 0.767 vs 0.704 (gap 0.063); year 0.998 vs 0.971 (gap 0.027)')}
MEETING07_EPOCHS = [(1, 7.27, 9.71), (2, 12.0, 10.68), (3, 16.36, 13.59), (4, 17.45, 14.56), (5, 21.45, 20.39), (6, 17.09, 19.42), (7, 16.36, 18.45), (8, 24.0, 23.3), (9, 22.55, 22.33), (10, 28.73, 30.1), (11, 31.64, 23.3), (12, 26.91, 23.3), (13, 27.27, 26.21), (14, 29.45, 28.16), (15, 32.36, 31.07), (16, 32.36, 31.07), (17, 29.09, 34.95), (18, 34.55, 30.1), (19, 32.36, 34.95), (20, 34.91, 42.72), (21, 33.09, 37.86), (22, 36.73, 35.92), (23, 30.91, 34.95), (24, 34.91, 36.89), (25, 36.73, 37.86)]


def render_meeting07():
    """Show original code excerpts and aggregate outputs alongside their scope."""
    st.title("FinIntel AI | Meeting 07")
    st.caption("Advisor requests and a separate SEC extraction pilot")
    st.info("These are development results. The planned independent test has not run.")
    requests_tab, pilot_tab, walkthrough_tab = st.tabs(["Requests 1–6", "Colab evidence · 36 companies", "Colab walkthrough · 27 steps"])

    def show_evidence(caption, code, output):
        st.markdown(f"**{caption}**")
        st.caption("Source code excerpt (not run by Streamlit)")
        st.code(code, language="python")
        st.caption("Saved output or verified audit result")
        st.code(output, language="text")

    def score_chart(rows, measure, denominator=None):
        data = pd.DataFrame(rows, columns=["Group", measure])
        encoding = {
            "y": {"field": "Group", "type": "nominal", "sort": None, "title": None},
            "x": {"field": measure, "type": "quantitative", "title": measure},
            "tooltip": [{"field": "Group"}, {"field": measure, "type": "quantitative"}],
        }
        if denominator is not None:
            encoding["x"]["scale"] = {"domain": [0, denominator]}
        st.vega_lite_chart(
            data,
            {
                "height": {"step": 27},
                "layer": [
                    {"mark": {"type": "bar", "color": "#176B87"}, "encoding": encoding},
                    {"mark": {"type": "text", "align": "right", "dx": -6, "color": "white"},
                     "encoding": {**encoding, "text": {"field": measure, "type": "quantitative", "format": ".2f" if any(isinstance(value, float) for _, value in rows) else "d"}}},
                ],
            },
            use_container_width=True,
        )

    measurements = {
        1: (
            [("Temporal selector", 105), ("Locked fusion", 105)],
            "Correct out of 115", 115,
            "Both counts remained unchanged in all 24 presentations of the same 115 questions.",
        ),
        2: (
            [("Model C", 57), ("Model D", 62)], "Correct out of 96", 96,
            "Same 96 Validation questions and fixed epoch 5. The paired difference was not statistically significant.",
        ),
        3: (
            [("Model C", 6), ("Model D", 4)], "Exact cells out of 32", 32,
            "Archived 224 px run, eight companies. The reported 336 px check had the same counts, but its exact runner was not recovered.",
        ),
        4: (
            [("Correct", 150), ("Answered", 162)], "Variants out of 192", 192,
            "Standalone four-choice rule only. In the separate option-free extraction task, C got 6/32 and D got 4/32.",
        ),
        5: (
            [("ICC .05", 93), ("ICC .10", 143), ("ICC .20", 242)],
            "Companies required", None,
            "Planning only: 25% paired discordance and 24 questions per company. The proposed target rounds 143 up to 144 companies.",
        ),
        6: (
            [("Model C", 96), ("Model D", 96)], "Archived choices matched out of 96", 96,
            "6B: replay agreement, not accuracy. 6A: zero tokens lost in 2,632 inputs (2,172 Train and 460 Validation).",
        ),
    }

    with requests_tab:
        st.caption("Each request has its original code or a relevant excerpt, its output, and what the result does and does not show.")
        for number, title, result, scope, blocks in MEETING07_EVIDENCE:
            st.subheader(f"Request {number} · {title}")
            st.markdown(f"**{result}**")
            st.write(scope)
            rows, measure, denominator, chart_note = measurements[number]
            score_chart(rows, measure, denominator)
            st.caption(chart_note)
            with st.expander(f"Request {number} · Code and output", expanded=(number == 1)):
                for caption, code, output in blocks:
                    show_evidence(caption, code, output)
        st.divider()
        st.download_button(
            "Download the Colab evidence notebook",
            MEETING07_NOTEBOOK,
            file_name="FinIntel_Meeting_7_Evidence_Excerpts.ipynb",
            mime="application/x-ipynb+json",
        )
        st.caption("The download contains the same public excerpts, not the private notebooks or datasets. The Request 5 power calculation can be run on its own in Colab.")
    with pilot_tab:
        st.subheader("A separate SEC pilot across 36 companies")
        st.caption("Separate from the earlier 96-question Validation set and the 32-question New8 check.")
        counts = st.columns(3)
        counts[0].metric("Train", "20 companies")
        counts[1].metric("Validation", "8 companies")
        counts[2].metric("Original test", "8 companies")
        st.caption("Step 7 · 480 questions after source checks: 275 Train, 103 Validation, 102 original test. No company was in two splits.")

        st.markdown("**1. Starting point: value-only baseline**")
        st.table([
            {"Model": model, "Correct cells (out of 102)": count}
            for model, count in (("A", 8), ("B", 37), ("C", 29), ("D", 31))
        ])
        score_chart([("Model A", 8), ("Model B", 37), ("Model C", 29), ("Model D", 31)], "Correct out of 102", 102)
        st.caption("The original 102-question panel. Step 26 checked these figures against the saved results.")
        st.markdown("**2. OCR recovered the printed headers**")
        st.write("Step 16: OCR read all 1,302 printed headers on the available original-test crops. This is transcription accuracy, not answer extraction accuracy.")
        with st.expander("Step 16 · OCR code and saved transcription output"):
            show_evidence("Original Colab transcription report", *MEETING07_OCR)
        st.caption("Step 11 · For value-only D, Validation peaked at 42.72% at epoch 20 and fell to 37.86% at epoch 25. Early stopping kept the epoch-20 checkpoint.")

        st.markdown("**3. Rechecked extraction after removing the answer-key date**")
        st.table([
            {"Model": "C", "Mean correct out of 102": 95.00},
            {"Model": "D", "Mean correct out of 102": 97.67},
        ])
        score_chart([("Model C", 95.0), ("Model D", 97.67)], "Mean correct out of 102", 102)
        st.write("Separate fresh panel, three seeds. D was ahead by 2.61 percentage points (95% company-cluster interval: −1.85 to +10.10). That does not establish that D is better. This fresh-panel result is not a before-and-after comparison with the original value-only baseline.")
        st.caption("Source: corrected rerun audit. PCA evaluation is in progress.")

        st.divider()
        st.subheader("What the original Colab stages actually showed")
        st.warning("Keep these archived stages separate from corrected results. Some used an answer-key date, and the ViT diagnostic gave the two models different inputs.")
        for caption, scope, code, output in MEETING07_PILOT_ARCHIVE:
            with st.expander(caption):
                st.write(scope)
                show_evidence("Original Colab code and recorded output", code, output)


    def render_walkthrough():
            st.subheader("Colab v4 · complete walkthrough")
            st.caption("27 steps across all 62 cells: 30 explanatory cells and 32 code cells. This view reads the existing notebook; it does not run new experiments.")
            st.info("Follow the steps in order. Code and saved outputs are shown separately from interpretation. The full private notebook keeps its existing access controls.")
            st.markdown("### Results highlights")
            st.markdown("**Step 16 · Verified OCR:** 1,302 of 1,302 printed headers read correctly on original-test crops. This is transcription, not answer accuracy.")
            st.markdown("**Corrected OCR rerun · Separate fresh panel:** C averaged 95.00 and D 97.67 correct out of 102 across three seeds (+2.61 percentage points; 95% CI −1.85 to +10.10). The difference does not establish that D is better.")
            st.table([
                {"Step": "19", "Result to review": "Original master table: old OCR jump (C 29 → 98 out of 102)", "How to read it": "Historical, affected by answer-key date leakage; not an OCR-only gain"},
                {"Step": "20", "Result to review": "Three-seed C vs. D on the original panel", "How to read it": "Historical OCR + ISO; affected by date leakage"},
                {"Step": "23", "Result to review": "Nine Model D gate variants", "How to read it": "Historical OCR + ISO; affected by date leakage"},
                {"Step": "25", "Result to review": "Frozen ViT vs. ViT + OCR-similarity gate", "How to read it": "Historical diagnostic; inputs were unequal"},
                {"Step": "26", "Result to review": "62 of 62 stored-report checks matched", "How to read it": "Archive verification only; no model rerun"},
            ])
            highlighted = {16, 19, 20, 23, 25, 26}
            stage_picks = st.selectbox("Jump to a notebook step", ["All 27 steps", "Highlighted result steps"] + [f"Step {n:02d} · {title}" for n, title, *_ in MEETING07_STAGES])
            for n, title, methods, results, scope, cells in MEETING07_STAGES:
                label = f"Step {n:02d} · {title}"
                if stage_picks == "Highlighted result steps" and n not in highlighted:
                    continue
                if stage_picks not in ("All 27 steps", "Highlighted result steps", label):
                    continue
                with st.expander(label, expanded=(stage_picks == label)):
                    st.markdown("**What was done**")
                    st.write(methods)
                    st.markdown("**What the saved notebook shows**")
                    st.write(results)
                    st.caption(scope)
                    st.dataframe(cells, hide_index=True)
                    if n in MEETING07_STAGE_EXCERPTS:
                        code, output = MEETING07_STAGE_EXCERPTS[n]
                        show_evidence("Selected primary evidence", code, output)
                    elif n == 16:
                        show_evidence("Saved OCR transcription", *MEETING07_OCR)
                    elif n in (19, 20, 23, 25, 26):
                        stage_record = next(row for row in MEETING07_PILOT_ARCHIVE if row[0].startswith(f"Step {n} ·"))
                        show_evidence("Selected archived output", stage_record[2], stage_record[3])
                    if n == 3:
                        score_chart([("Before cleaning", 465), ("After cleaning", 480)], "Facts found out of 540", 540)
                    elif n == 8:
                        st.line_chart(pd.DataFrame({"Top k": [1, 3, 5, 10, 15], "Retrieval recall (%)": [21.57, 57.84, 72.55, 95.10, 100.00]}), x="Top k", y="Retrieval recall (%)")
                    elif n == 11:
                        epochs = pd.DataFrame(MEETING07_EPOCHS, columns=["Epoch", "Train (%)", "Validation (%)"])
                        st.line_chart(epochs, x="Epoch", y=["Train (%)", "Validation (%)"])
            with st.expander("Full notebook index · all 62 cells"):
                st.dataframe([cell for *_, cells in MEETING07_STAGES for cell in cells], hide_index=True)
            st.subheader("Next Steps")
            st.write("PCA evaluation is in progress.")

    with walkthrough_tab:
        render_walkthrough()

dashboard_view = st.sidebar.radio(
    "FinIntel AI Research Dashboard",
    [
        "Meeting 02 — Pipeline Feasibility",
        "Meeting 03 — Learning & Evaluation",
        "Meeting 04 — Multimodal Grounding & Fusion",
        "Meeting 05 — Grounding Audit & Diagnostics",
        "Meeting 06 — Readable Evidence & Next Steps",
        "Meeting 07 — Controlled Evaluation",
    ],
    index=5 if st.query_params.get("meeting") == "7" else 4,
)

st.sidebar.caption(
    "Meeting 02 and Meeting 03 preserve earlier research history. "
    "Meeting 04 is an independent, research-scale experiment: "
    "30 companies × 5 fiscal years × 6 financial concepts. "
    "Meeting 05 separates neural diagnostics from rule-based source extraction."
)

if dashboard_view == "Meeting 07 — Controlled Evaluation":
    render_meeting07()
    st.stop()

if dashboard_view == "Meeting 06 — Readable Evidence & Next Steps":
    render_meeting06()
    st.stop()

if dashboard_view == "Meeting 05 — Grounding Audit & Diagnostics":
    render_meeting05()
    st.stop()

if dashboard_view == "Meeting 04 — Multimodal Grounding & Fusion":
    render_meeting04()
    st.stop()
# ============================================================
# Meeting 03 routing
# ============================================================

if dashboard_view == "Meeting 03 — Learning & Evaluation":
    try:
        meeting03_evidence, meeting03_summary_data, meeting03_checks = (
            load_and_validate_meeting03()
        )
    except Meeting03EvidenceError as exc:
        st.error("Meeting 03 evidence package could not be loaded.")
        st.code(str(exc))
        st.stop()

    st.title("FinIntel AI — Meeting 03")
    st.subheader("Learning & Evaluation")

    st.info(
        "Meeting 03 extends the validated POC from technical pipeline "
        "feasibility to supervised learning and temporal held-out evaluation."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Labeled Samples",
        meeting03_summary_data["samples"],
    )
    c2.metric(
        "Train",
        meeting03_summary_data["train_samples"],
    )
    c3.metric(
        "Held-out",
        meeting03_summary_data["heldout_samples"],
    )
    c4.metric(
        "10-K Filings",
        meeting03_summary_data["filings"],
    )
    c5.metric(
        "XBRL Concepts",
        meeting03_summary_data["concepts"],
    )

    st.divider()

    st.caption(
        "Meeting 03 evidence package loaded and validated successfully."
    )

    # Stop here so the original Meeting 02 dashboard below
    # remains unchanged and runs only when Meeting 02 is selected.
    # ============================================================
    # Meeting 03 — Evaluation Results
    # ============================================================
    meeting03_evidence, meeting03_summary_data, meeting03_checks = (
        load_and_validate_meeting03()
    )
    st.subheader("Temporal Held-Out Evaluation")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Concept Micro-F1",
        format(meeting03_summary_data["concept_micro_f1"], ".3f"),
    )

    r2.metric(
        "Value Exact Match",
        format(meeting03_summary_data["value_exact_match"], ".3f"),
    )

    r3.metric(
        "Joint Extraction",
        format(meeting03_summary_data["joint_extraction"], ".3f"),
    )

    r4.metric(
        "Historical Retrieval P@5",
        "0.840",
    )


    st.caption(
        "Concept, value, joint extraction, and retrieval are separate evaluation quantities. "
        "Retrieval P@5 is the historical Meeting 02 retrieval result and is not extraction accuracy."
    )

    st.divider()

    # ============================================================
    # Current Research Finding
    # ============================================================

    st.subheader("Current Pilot Finding")

    st.success(
        "Concept identification succeeded on all six FY2025 temporal held-out samples "
        "(Micro-F1 = 1.000)."
    )

    st.warning(
        "Financial-value grounding remains the current bottleneck: "
        "Value Exact Match = 0.167 and Joint Extraction = 0.167."
    )

    st.info(
        "The current pilot does not establish multimodal superiority or "
        "company-level generalization."
    )

    st.divider()

    # ============================================================
    # Error Analysis
    # ============================================================

    st.subheader("Held-Out Error Analysis")

    e1, e2, e3 = st.columns(3)

    e1.metric("Held-Out Samples", 6)
    e2.metric("Correct Joint Extractions", 1)
    e3.metric("Value-Selection Errors", 5)

    st.markdown(
        """
    **Observed failure modes**

    - **3 / 5 errors:** year / column confusion
    - **2 / 5 errors:** nearby row / metric confusion
    - **6 / 6 predictions:** selected numeric value was present in the corresponding SEC evidence
    """
    )

    st.caption(
        "Interpretation: the dominant failure is value grounding/selection, "
        "rather than absence of the target information from the retrieved SEC evidence."
    )

    st.divider()

    # ============================================================
    # Meeting 03 research boundary
    # ============================================================

    st.subheader("Research Boundary")

    st.markdown(
        """
    **Meeting 03 supports**

    - supervised training of prediction/fusion components
    - SEC/XBRL-grounded targets
    - temporal held-out pilot evaluation
    - initial six-condition model comparison
    - failure-mode identification
    - statistical testing and power-analysis planning

    **Meeting 03 does not yet claim**

    - company-level generalization
    - statistically significant multimodal superiority
    - final H1 / H2 / H3 confirmation
    - a power-derived final test-set size
    """
    )

    # Stop before the original Meeting 02 application.
    # ============================================================
    # Meeting 03 — Requirement Evidence / History
    # ============================================================

    st.divider()

    st.subheader("Meeting 03 Requirement Evidence")
    st.caption(
        "Traceability from advisor requirements to the evidence generated "
        "during the Meeting 03 learning and evaluation cycle."
    )

    requirement_groups = [
        {
            "group": "Requirements 01–03",
            "title": "Task Definition & Ground Truth",
            "status": "✅ Complete",
            "evidence": (
                "Exact extraction task defined; value and XBRL concept objectives "
                "specified; SEC/XBRL-grounded labeling and verification protocol established."
            ),
        },
        {
            "group": "Requirements 04–07",
            "title": "Training & Held-Out Evaluation",
            "status": "✅ Complete",
            "evidence": (
                "Prediction/fusion components trained with backpropagation; "
                "FY2025 temporal held-out evaluation completed; Precision/Recall/F1 "
                "and value/joint extraction results reported; failure modes analyzed."
            ),
        },
        {
            "group": "Requirements 08–11",
            "title": "Dataset, Pairing & Leakage Control",
            "status": "✅ Complete",
            "evidence": (
                "40 labeled samples across five Apple 10-K filings; 40/40 valid "
                "text–SEC-table pairs; broader 10-K multimodal scope defined; "
                "temporal leakage controls documented."
            ),
        },
        {
            "group": "Requirements 12–13",
            "title": "Model Comparison & Reproducibility",
            "status": "✅ Complete",
            "evidence": (
                "Six controlled model conditions evaluated using the same temporal "
                "held-out set; parameter counts, optimizer, learning rate, loss, "
                "epochs, encoders, and random seed documented."
            ),
        },
        {
            "group": "Requirements 14–15",
            "title": "Retrieval vs. Extraction",
            "status": "✅ Complete",
            "evidence": (
                "Retrieval relevance, concept classification, value extraction, "
                "and joint extraction separated as distinct evaluation quantities; "
                "retrieval relevance rules and future query-level logging protocol defined."
            ),
        },
        {
            "group": "Requirements 16–17",
            "title": "Statistical Validation & Power Planning",
            "status": "✅ Complete",
            "evidence": (
                "Primary paired comparison and Micro-F1 outcome pre-specified; "
                "95% confidence interval and group-aware resampling framework defined; "
                "alpha = 0.05 and target power = 0.80 established for final test-size planning."
            ),
        },
       {
        "group": "Final Evidence Package",
        "title": "Meeting 03 Final Evidence Package",
        "status": "✅ Complete",
        "evidence": (
            "Final evidence summary, Completed-vs-WIP matrix, key findings, "
            "research boundary, Streamlit dashboard, final PowerPoint, "
            "and final checkpoint completed."
        ),
    },
    ]

    for item in requirement_groups:
        with st.expander(
            f"{item['group']} — {item['title']} — {item['status']}",
            expanded=False,
        ):
            st.write(item["evidence"])

    st.info(
        "Meeting 03 advances FinIntel AI from pipeline feasibility to supervised "
        "learning and temporal held-out evaluation. Final hypothesis conclusions "
        "remain deferred until the expanded, company-diverse evaluation."
    )
    st.stop()


# Legacy Meeting 02 alone requires its original checkpoint and ML dependencies.
try:
    from src.data_loader import (POCDataLoadError, compute_retrieval_metrics,
                                 load_poc_data, run_pipeline_validation)
    from src.retrieval import build_faiss_index, live_search
except ImportError as error:
    st.error("Meeting 02 requires its original full environment. See README_REVIEW.md.")
    st.code(str(error))
    st.stop()

# ----------------------------------------------------------------------
# Data loading (cached for the process lifetime)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading verified POC checkpoint...")
def get_data():
    return load_poc_data()


try:
    data = get_data()
except POCDataLoadError as e:
    st.error(
        "**Could not load the POC checkpoint.**\n\n"
        f"{e}\n\n"
        "This app only displays real, checkpointed POC outputs. It will not "
        "start with synthetic substitutes."
    )
    st.stop()

checks, all_passed = run_pipeline_validation(data)
retrieval_metrics = compute_retrieval_metrics(data)
faiss_index = build_faiss_index(data.embeddings)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("FinIntel AI -- Multimodal Financial Intelligence POC")
st.caption("Proof of Concept -- Technical Feasibility Demonstration")
st.warning(
    "This POC demonstrates technical pipeline feasibility. Results are "
    "preliminary and do not establish H1, H2, or H3.",
    icon="⚠️",
)

st.markdown(
    "**Pipeline:** SEC 10-K &rarr; Evidence Retrieval &rarr; Text Representation "
    "(FinBERT) &rarr; Visual Representation (ViT) &rarr; Bidirectional "
    "Cross-Attention &rarr; Adaptive Fusion &rarr; POC Output / Validation"
)

# ----------------------------------------------------------------------
# Sidebar -- selection
# ----------------------------------------------------------------------
st.sidebar.header("Selection")

corpus_df = pd.DataFrame(
    [{"ticker": r["ticker"], "accession": r["accession"]} for r in data.item7_corpus]
).sort_values(["ticker", "accession"])
corpus_df["label"] = corpus_df["ticker"] + " -- " + corpus_df["accession"]

filing_label = st.sidebar.selectbox(
    "MD&A filing (15-filing retrieval corpus)",
    corpus_df["label"].tolist(),
    help="Covers all 5 companies in the retrieval pilot: AAPL, AMZN, GOOGL, MSFT, NVDA.",
)
sel_ticker, sel_accession = filing_label.split(" -- ")

pairs_df = data.clean_multimodal_df.sort_values("pair_id").reset_index(drop=True)
pair_labels = [
    f"Pair {row.pair_id} -- {row.ticker} -- {row.accession}"
    for row in pairs_df.itertuples()
]
pair_label = st.sidebar.selectbox(
    "Multimodal pair (chart + MD&A, 7 clean pairs)",
    pair_labels,
    help="Only AAPL (3), GOOGL (3), and NVDA (1) have a validated chart+text pair "
    "in the current POC subset. AMZN and MSFT have none.",
)
sel_pair_idx = pair_labels.index(pair_label)
sel_pair = pairs_df.iloc[sel_pair_idx]

st.sidebar.divider()
st.sidebar.caption(
    f"Companies with a visual pair: {', '.join(sorted(pairs_df['ticker'].unique()))}\n\n"
    "No visual pair available for AMZN or MSFT in the current POC subset."
)


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
(
    tab_overview,
    tab_filing,
    tab_retrieval,
    tab_visual,
    tab_repr,
    tab_fusion,
    tab_validation,
    tab_boundary,
) = st.tabs(
    [
        "Overview",
        "Filing & MD&A",
        "Retrieval / RAG",
        "Visual Evidence",
        "Text & Visual Representations",
        "Cross-Attention & Adaptive Fusion",
        "POC Validation",
        "Research Boundary",
    ]
)

# --- Overview -----------------------------------------------------------
with tab_overview:
    st.subheader("What this POC demonstrates")
    c1, c2, c3 = st.columns(3)
    c1.metric("MD&A filings", len(data.item7_corpus))
    c1.metric("Companies (retrieval)", corpus_df["ticker"].nunique())
    c2.metric("Retrieval chunks", len(data.item7_chunks))
    c2.metric("Clean multimodal pairs", len(data.clean_multimodal_df))
    c3.metric("Pipeline checks passed", f"{sum(checks.values())}/{len(checks)}")
    c3.metric("Mean Precision@5 (pilot)", f"{retrieval_metrics['mean_precision_at_5']:.3f}")

    st.markdown(
        """
        **Architecture:** SEC 10-K narrative text and financial visuals are
        independently encoded (FinBERT for text, ViT for charts), combined
        through bidirectional cross-attention, and combined again through an
        adaptive modality-weighting gate into a single fused representation.
        A FAISS-indexed retrieval layer surfaces relevant MD&A evidence
        before any multimodal processing occurs.
        """
    )
    st.caption(
        "Use the sidebar to pick an MD&A filing (drives the Filing/Retrieval "
        "tabs) and a multimodal pair (drives the Visual/Representation/Fusion "
        "tabs) -- the two selectors cover different subsets of the data, by "
        "design (see Research Boundary tab)."
    )

# --- Filing & MD&A --------------------------------------------------------
with tab_filing:
    record = next(
        r
        for r in data.item7_corpus
        if r["ticker"] == sel_ticker and r["accession"] == sel_accession
    )
    st.subheader(f"{sel_ticker} -- {sel_accession}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Section", record["section"])
    m2.metric("Characters extracted", f"{record['chars']:,}")
    m3.metric("End boundary", record["end_status"])

    n_chunks = sum(
        1
        for c in data.item7_chunks
        if c["ticker"] == sel_ticker and c["accession"] == sel_accession
    )
    st.caption(f"This filing contributes {n_chunks} chunks to the 515-chunk retrieval index.")

    with st.expander("Item 7 / MD&A text (extracted)", expanded=False):
        st.text(record["text"][:6000])
        if len(record["text"]) > 6000:
            st.caption(f"Showing first 6,000 of {len(record['text']):,} characters.")

# --- Retrieval / RAG ------------------------------------------------------
with tab_retrieval:
    st.subheader("Evidence Retrieval")
    st.caption(
        "FAISS cosine-similarity search (IndexFlatIP over normalized MiniLM "
        "embeddings, all-MiniLM-L6-v2) rebuilt at startup from the 515 "
        "verified chunk embeddings."
    )

    mode = st.radio(
        "Query source",
        ["5 evaluated queries (manually judged)", "Custom query (live retrieval)"],
        horizontal=True,
    )

    if mode.startswith("5 evaluated"):
        q = st.selectbox("Evaluated query", retrieval_metrics["queries"])
        subset = (
            data.retrieval_eval_df[data.retrieval_eval_df["query"] == q]
            .sort_values("rank")
        )
        p_at_5 = retrieval_metrics["precision_at_5_by_query"][q]
        st.success(
            f"Manually evaluated -- Precision@5 for this query: **{p_at_5:.2f}** "
            f"({int(p_at_5 * 5)}/5 relevant)"
        )
        show_df = subset[["rank", "ticker", "accession", "score", "relevant", "text"]].copy()
        show_df["relevant"] = show_df["relevant"].map({1: "Relevant", 0: "Not relevant"})
        show_df["text"] = show_df["text"].str.slice(0, 220) + "..."
        st.dataframe(show_df, width="stretch", hide_index=True)

        st.info(
            f"Preliminary Retrieval Pilot (all 5 queries): "
            f"**{retrieval_metrics['relevant_count']}/{retrieval_metrics['total_judged']} relevant**, "
            f"Mean Precision@5 = **{retrieval_metrics['mean_precision_at_5']:.3f}**.\n\n"
            "Preliminary manual POC retrieval evaluation -- not a confirmatory "
            "research metric."
        )

    else:
        st.markdown(
            "**Live POC Retrieval -- Not Manually Evaluated.** Results below "
            "come from a live FAISS search and carry no relevance labels. "
            "They do not contribute to or alter the reported "
            f"Mean Precision@5 = {retrieval_metrics['mean_precision_at_5']:.3f}, "
            "which is fixed from the 25 manually-judged results above."
        )
        custom_query = st.text_input(
            "Custom query", placeholder="e.g. How did AI infrastructure spending affect margins?"
        )
        if st.button("Search", type="primary") and custom_query:
            with st.spinner("Encoding query and searching..."):
                results, error = live_search(custom_query, faiss_index, data.item7_chunks)

            if error:
                st.error(
                    f"Live retrieval unavailable: {error}\n\n"
                    "Falling back to the evaluated-query demonstration -- "
                    "select '5 evaluated queries' above to continue the demo."
                )
            else:
                st.caption("Live POC Retrieval -- Not Manually Evaluated")
                res_df = pd.DataFrame(results)
                res_df["text"] = res_df["text"].str.slice(0, 220) + "..."
                st.dataframe(res_df, width="stretch", hide_index=True)

# --- Visual Evidence --------------------------------------------------------
with tab_visual:
    st.subheader(f"Pair {sel_pair.pair_id} -- {sel_pair.ticker} -- {sel_pair.accession}")
    image_candidates = list(ASSETS_DIR.glob(f"pair{sel_pair.pair_id}_*"))
    if image_candidates:
        st.image(
            str(image_candidates[0]),
            caption=f"{sel_pair.filename} ({sel_pair.semantic_type})",
            width=650,
        )
        st.caption(
            "Source: real SEC 10-K financial chart, extracted and verified in "
            "the source POC notebook."
        )
    else:
        st.warning("No visual pair available in the current POC subset.")

# --- Text & Visual Representations ------------------------------------------
with tab_repr:
    st.subheader("Text Representation -- FinBERT")
    c1, c2 = st.columns(2)
    c1.metric("Token sequence length", data.text_tokens.shape[1])
    c1.metric("Representation dimension", data.text_tokens.shape[2])
    c2.metric("Attention mask length", data.text_masks.shape[1])
    c2.metric("Pairs encoded", data.text_tokens.shape[0])
    st.caption("Model: ProsusAI/finbert. Token-level hidden states, not a single pooled vector.")

    st.divider()

    st.subheader("Visual Representation -- ViT")
    c3, c4 = st.columns(2)
    c3.metric("Patch tokens", data.visual_patches.shape[1])
    c3.metric("Representation dimension", data.visual_patches.shape[2])
    c4.metric("Pairs encoded", data.visual_patches.shape[0])
    st.caption("Model: google/vit-base-patch16-224-in21k. CLS token excluded from patch tensor.")

# --- Cross-Attention & Adaptive Fusion ---------------------------------------
with tab_fusion:
    # pairs_df preserves the original checkpoint row order (already sorted by
    # pair_id in the source data), which matches the tensor batch order used
    # when the 7 pairs were encoded -- so sel_pair_idx indexes both correctly.
    i = sel_pair_idx

    st.subheader("Bidirectional Cross-Attention")
    st.markdown(
        "Text and visual token sequences attend to each other in both "
        "directions before being pooled and concatenated."
    )
    fused_shape = tuple(data.cross_attention_fused[i : i + 1].shape)
    st.metric(f"Cross-attention fused representation -- pair {sel_pair.pair_id}", str(fused_shape))

    st.divider()

    st.subheader("Adaptive Fusion")
    st.markdown(
        """
        ```
        Text
           \\
            Adaptive Fusion --> Fused Representation
           /
        Visual
        ```
        """
    )
    w = data.modality_weights[i]
    wc1, wc2 = st.columns(2)
    wc1.metric("Text weight", f"{w[0]:.3f}")
    wc2.metric("Visual weight", f"{w[1]:.3f}")
    st.warning(
        "POC / untrained gate outputs -- not learned modality importance. "
        "These weights come from an untrained adaptive gate and must not be "
        "read as evidence that text and visual modalities are equally "
        "important, or as a research finding.",
        icon="⚠️",
    )

    single_shape = tuple(data.adaptive_fused_repr[i : i + 1].shape)
    full_shape = tuple(data.adaptive_fused_repr.shape)
    st.markdown(
        f"**This pair's fused representation:** `{single_shape}` "
        f"(one of the {full_shape[0]} POC pairs)\n\n"
        f"**Full seven-pair POC fused representation:** `{full_shape}`"
    )

# --- POC Validation --------------------------------------------------------
with tab_validation:
    st.subheader("Technical Validation")
    st.caption(
        "Recomputed live against the loaded checkpoint -- not a hardcoded string."
    )
    for name, passed in checks.items():
        st.markdown(f"{'✓' if passed else '✗'} {name}")
    if all_passed:
        st.success(f"{sum(checks.values())}/{len(checks)} pipeline validation checks passed.")
    else:
        st.error(f"{sum(checks.values())}/{len(checks)} pipeline validation checks passed -- review above.")

    st.divider()
    st.subheader("Preliminary Retrieval Pilot")
    st.markdown(
        f"**{retrieval_metrics['relevant_count']} / {retrieval_metrics['total_judged']} relevant** &mdash; "
        f"Mean Precision@5 = **{retrieval_metrics['mean_precision_at_5']:.3f}**"
    )
    st.caption(
        "Preliminary manual POC retrieval evaluation -- not a confirmatory research metric."
    )

# --- Research Boundary -------------------------------------------------------
with tab_boundary:
    st.subheader("What this POC demonstrates")
    st.markdown(
        """
- Pipeline integration
- Model compatibility (FinBERT + ViT + cross-attention + adaptive fusion)
- Tensor / representation compatibility
- Retrieval feasibility (FAISS over real SEC MD&A text)
- Multimodal fusion execution
- Technical feasibility, end to end
        """
    )
    st.subheader("What this POC does NOT yet demonstrate")
    st.markdown(
        """
- H1 superiority over a text-only baseline
- H2 superiority over simple concatenation
- H3 statistically significant multimodal gains
- Final research performance
- Learned modality importance (the fusion gate is untrained)
        """
    )
    split = data.status.get("pilot_split", {})
    st.info(
        f"Current multimodal subset: {len(data.clean_multimodal_df)} pairs total "
        f"({split.get('train', '?')} train / {split.get('holdout', '?')} holdout) "
        "-- far too small for a performance claim. This subset exists to validate "
        "that the pipeline runs end to end, not to evaluate it."
    )


