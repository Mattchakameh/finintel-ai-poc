"""Meeting 6: concise results first, with evidence and limits one click away."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.meeting06_data import EvidenceError, load_evidence, stability

ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = {"native": "Original image", "model224": "Resized image (224 × 224)"}
SPLIT_NAMES = {
    "Train": "Train · 543 questions",
    "Validation": "Validation · 115 questions",
    "Test": "Historical Test · 130 questions · previously exposed",
}


def render_meeting06():
    st.title("FinIntel AI | Meeting 06")
    st.caption(
        "Readable evidence, honest comparisons, clear next steps · Review copy · 12 November 2026"
    )
    try:
        evidence = load_evidence()
    except EvidenceError as error:
        st.error("Meeting 06 evidence unavailable. No substitute results are shown.")
        st.code(str(error))
        return
    meta = evidence["meta"]
    st.info(
        "Main finding: Qwen selects the original four-choice answer much more often from readable "
        "original table images than from the same tables reduced to 224×224. This is a diagnostic "
        "result, not a demonstrated improvement to the FinBERT/ViT model."
    )
    st.success(
        "New learned-extraction result: D selects 62/96 cells correctly (64.58%), "
        "versus C 57/96, B 41/96 and the original reference 56/96. "
        "The saved-run archive was audited, not independently neural-replayed. See the B/C/D results tab."
    )
    st.caption(
        "Two different studies: Qwen is a frozen four-choice image diagnostic; B/C/D are trained cell-selection heads. Their percentages are not directly comparable."
    )
    st.subheader("Original image vs. resized image (224 × 224)")
    st.caption("Original means the source image before Qwen internal preprocessing; both inputs subsequently pass through Qwen’s image processor.")
    split = st.selectbox(
        "Show Qwen results for",
        list(SPLIT_NAMES),
        index=1,
        format_func=SPLIT_NAMES.get,
        key="m06_split",
    )
    if split == "Test":
        st.warning(
            "Historical Test has already been exposed. These results are descriptive, not an untouched holdout or a confirmatory generalization test."
        )
    elif split == "Train":
        st.caption(
            "Original Train partition. Qwen is frozen; it was not trained on this run. This is not held-out evidence."
        )
    else:
        st.caption(
            "Reused Validation development set. The original 115 questions and labels are retained."
        )
    selected = {r["arm"]: r for r in evidence["results"] if r["split"] == split}
    native, reduced = selected["native"], selected["model224"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Original image · mean accuracy", f"{native['mean']:.2f}%")
    c1.caption(f"± {native['sd']:.2f} percentage points across option orders")
    c2.metric("Resized image (224 × 224) · mean accuracy", f"{reduced['mean']:.2f}%")
    c2.caption(f"± {reduced['sd']:.2f} percentage points across option orders")
    c3.metric("Original minus resized", f"+{native['mean'] - reduced['mean']:.2f} pp")
    c3.caption(f"Same {native['n']} questions · 5 option orders · 25% chance")
    st.caption(
        "Mean ± sample SD across presentation seeds 11, 22, 33, 44 and 55. "
        "These are not five training runs or five independent datasets."
    )
    chart_data = pd.DataFrame(
        [
            {
                "Image": ARM_NAMES[r["arm"]],
                "Accuracy (%)": r["mean"],
                "low": max(0, r["mean"] - r["sd"]),
                "high": min(100, r["mean"] + r["sd"]),
            }
            for r in [native, reduced]
        ]
    )
    bars = (
        alt.Chart(chart_data)
        .mark_bar(size=42)
        .encode(
            y=alt.Y("Image:N", sort=["Original image", "Resized image (224 × 224)"], title=None),
            x=alt.X("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Image:N",
                scale=alt.Scale(
                    domain=["Original image", "Resized image (224 × 224)"],
                    range=["#147d92", "#8e99a8"],
                ),
                legend=None,
            ),
            tooltip=["Image", "Accuracy (%)"],
        )
    )
    error = (
        alt.Chart(chart_data)
        .mark_rule(color="#26364a", strokeWidth=2)
        .encode(y="Image:N", x="low:Q", x2="high:Q")
    )
    chance = (
        alt.Chart(pd.DataFrame({"chance": [25]}))
        .mark_rule(strokeDash=[4, 4], color="#b36b00")
        .encode(x="chance:Q")
    )
    st.altair_chart((bars + error + chance).properties(height=160), width="stretch")
    st.caption(
        "Whiskers show variation across option orders, not confidence intervals. Dashed line: four-choice chance (25%)."
    )
    overview, example_tab, diagnostic_tab, plan_tab, audit_tab = st.tabs(
        [
            "What this means",
            "See the evidence",
            "Reproducibility & next steps",
            "B/C/D verified results",
            "Scores & verification",
        ]
    )
    with overview:
        left, right = st.columns(2)
        with left:
            st.subheader("Supported by this result")
            st.write(
                "Readable image detail matters in this Qwen four-choice setup. The original-image result is consistently higher across all five option orders."
            )
            st.write(
                "7,880 saved responses were independently re-parsed and mapped back to the original option identities. All 30 score groups match. Qwen inference was not independently rerun."
            )
        with right:
            st.subheader("Still open")
            st.write(
                "The new B/C/D comparison is verified on reused Validation, but it is not a final model. A remains untrained because narrative-only supervision is not verified."
            )
            st.write(
                "A fresh, source-verified holdout is still needed for a generalization claim. Original source and label defects remain in this diagnostic."
            )
        st.warning(
            "Do not combine this score with extraction, retrieval, parser, or span accuracy. They answer different questions."
        )
    with example_tab:
        st.subheader("Same question, same table, two image conditions")
        st.caption(
            "Two deliberately selected Validation examples from option order 11: one original-image success and one original-image failure. They illustrate behavior, not prevalence. Correctness uses the retained original label, not new source adjudication."
        )
        case_index = st.selectbox(
            "Illustrative example",
            range(len(evidence["examples"])),
            format_func=lambda i: evidence["examples"][i]["label"],
            key="m06_case",
        )
        case = evidence["examples"][case_index]
        st.write(case["question"])
        opts = {x["id"]: x["value"] for x in case["options"]}
        mapping = case["rows"]["native"]["letter_to_option_id"]
        st.write(
            "Options (raw base units): "
            + "  ·  ".join(
                f"{letter}: {int(opts[oid]):,}" for letter, oid in mapping.items()
            )
        )
        for col, arm in zip(st.columns(2), ["native", "model224"]):
            row = case["rows"][arm]
            with col:
                st.markdown(f"**{ARM_NAMES[arm]}**")
                st.image(
                    str(ROOT / case["images"][arm]),
                    width="stretch" if arm == "native" else 224,
                )
                correct = row["option_id"] == case["gold_option_id"]
                st.write(
                    f"Saved response: {row['raw']} · {'matches' if correct else 'does not match'} original label"
                )
                st.caption(
                    f"Input tokens: {row['input_tokens']:,} · processor grid {row['image_grid_thw']}"
                )
        st.caption(
            "Both images then pass through the same Qwen processor pixel bounds. The Resized image (224 × 224) may be upsampled; lost detail is not restored. Token and compute budgets are not matched."
        )
        with st.expander("Original label and saved responses"):
            st.write(
                f"Original reference amount: {int(opts[case['gold_option_id']]):,} raw base units"
            )
            st.json(case["rows"])
    with diagnostic_tab:
        st.subheader("What independently reproduced, and what did not")
        st.caption(
            "Question/header study · three saved checkpoints · frozen input features · same 96 reused-Validation questions"
        )
        replay, training = evidence["header_replay"], evidence["header_training"]
        rows = []
        names = {
            "baseline": "Baseline",
            "pair_concat": "Question/header concatenation",
            "pair_interaction": "Question/header interaction",
        }
        for arm, name in names.items():
            r = replay["arms"][arm]
            fresh = training["fresh_results"][arm]["Validation"]
            repeat = training["repeat_results"][arm]["Validation"]
            rows.append(
                {
                    "Condition": name,
                    "Archived checkpoint replay": f"{r['correct']}/{r['denominator']}",
                    "Fresh training run 1": f"{fresh['correct']}/{fresh['denominator']}",
                    "Fresh training run 2": f"{repeat['correct']}/{repeat['denominator']}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.success(
            "All 288 archived checkpoint choices matched in independent inference replay: 53/96, 84/96 and 79/96. This replay uses frozen features; it does not regenerate the encoders."
        )
        st.warning(
            "The archived full training result did not reproduce. Two new training runs agreed with each other at 56/96, 82/96 and 82/96, but differed from the archived run. The cause is unresolved."
        )
        st.caption(
            "Replay logits are not bit-for-bit identical. Maximum absolute difference: 0.0001526. Numerical tolerance was reported after exact equality failed, not prespecified as a research outcome."
        )
        with st.expander("Evidence levels used throughout this dashboard"):
            st.write(
                "Qwen: original saved responses re-parsed and rescored, not independent Qwen inference."
            )
            st.write(
                "B/C/D: original saved-run archive and metrics audited, not an independent neural or training replay."
            )
            st.write(
                "Question/header: actual independent checkpoint inference with frozen features; full historical training reproduction failed."
            )
            st.write(
                "Earlier saved-only global-image, open-answer and span-demo results are omitted from this Meeting 06 review to match the final presentation. Older meeting pages remain historical records."
            )
        st.subheader("Next research steps")
        st.write("1. Establish direct extraction from the supplied evidence.")
        st.write(
            "2. Isolate narrative contribution with a controlled with/without-narrative comparison."
        )
        st.write(
            "3. Test image readability under matched C/D inputs and capacity controls."
        )
        st.write(
            "4. Freeze the method, then evaluate on an independent extraction test."
        )
        st.info(
            "H1, H2 and H3 remain unconfirmed. This review does not declare a final model."
        )
    with plan_tab:
        st.subheader("Learned extraction: verified B/C/D results")
        st.caption(
            "Same 96 reused-Validation questions · one training seed · fixed final epoch 5 · no Test evaluation"
        )
        bcd = evidence["bcd"]
        labels = {
            "original_reference": "Original reference",
            "B": "B · text + narrative",
            "C": "C · concatenation",
            "D": "D · attention + gate",
        }
        columns = st.columns(4)
        for column, arm in zip(columns, labels):
            result = bcd["metrics"][arm]["Validation"]
            column.metric(labels[arm], f"{result['correct']}/{result['denominator']}")
            column.caption(
                f"{100 * result['exact_cell_accuracy']:.2f}% exact cell selection"
            )
        pair = bcd["paired_validation"]["D_vs_C"]
        st.write(
            f"D versus C: {pair['fixes']} errors fixed, {pair['regressions']} new errors, "
            f"a net gain of {pair['net_correct']} correct answers on these 96 questions."
        )
        reference_pair = bcd["paired_validation"]["D_vs_original_reference"]
        st.write(
            f"D versus the original reference: {reference_pair['fixes']} fixes and "
            f"{reference_pair['regressions']} regressions, net +{reference_pair['net_correct']}."
        )
        st.warning(
            "This is an exploratory development result, not a final model or a generalization claim. "
            "Validation has been reused, only one training seed was run, C and D have different "
            "parameter counts, and there is no with/without-narrative ablation."
        )
        st.subheader("What the models actually receive")
        definitions = meta["bcd"]
        methods = [
            {
                "Arm": "Original reference",
                "Inputs and fusion": bcd["original_reference"],
                "Trainable parameters": f"{bcd['parameter_counts']['original_reference']:,}",
            }
        ]
        methods += [
            {
                "Arm": a,
                "Inputs and fusion": definitions[a],
                "Trainable parameters": f"{bcd['parameter_counts'][a]:,}"
                if a != "A"
                else "Not trained",
            }
            for a in ["A", "B", "C", "D"]
        ]
        st.dataframe(pd.DataFrame(methods), hide_index=True, width="stretch")
        st.info(
            "A remains NOT TRAINED: verified narrative-only supervision is unavailable. "
            "A was not replaced with table text or with the original reference model."
        )
        st.subheader("Narrative availability is still unresolved")
        availability = bcd["actual_neural_top2_number_presence_diagnostic"]
        st.write(
            f"The two retrieved narrative passages contained the literal printed target number in "
            f"{availability['Train']['printed_number_present']}/{availability['Train']['n']} Train and "
            f"{availability['Validation']['printed_number_present']}/{availability['Validation']['n']} Validation cases."
        )
        st.caption(
            "This checks number presence with comma/decimal/sign normalization only, without scale conversion. "
            "It is not semantic retrieval accuracy, does not establish uselessness, and does not show that narrative extraction succeeded."
        )
        with st.expander("Training, loss, and reproducibility"):
            table = []
            for arm, label in labels.items():
                train, val = (
                    bcd["metrics"][arm]["Train"],
                    bcd["metrics"][arm]["Validation"],
                )
                table.append(
                    {
                        "Arm": label,
                        "Train correct": f"{train['correct']}/537",
                        "Validation correct": f"{val['correct']}/96",
                        "Validation loss (CE)": round(val["mean_ce"], 3),
                    }
                )
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            st.write(
                "D has higher exact-cell accuracy than C, but C has lower Validation cross-entropy loss "
                f"({bcd['metrics']['C']['Validation']['mean_ce']:.3f} versus {bcd['metrics']['D']['Validation']['mean_ce']:.3f}). "
                "D is not better on every measure."
            )
            st.write(
                f"537 Train, 96 eligible Validation, 19 excluded Validation records retained. "
                f"Seed {bcd['seed']}; {bcd['epochs']} fixed epochs; AdamW learning rate 0.001; batch 8; no dropout. "
                "Same schedule, fixed final checkpoint, no Validation-based checkpoint selection."
            )
            st.write("Loss: " + bcd["loss"] + ". Decoder: " + bcd["decoder"] + ".")
            st.caption(
                "Saved checkpoint bytes, recorded reload checks, input/provenance hashes and paired metrics were audited. This was not an independent neural replay. "
                "This dashboard displays the derived audit report; it does not bundle the 65 MB checkpoint archive "
                "or rerun training. The archive remains in the owner's saved files."
            )
        with st.expander("Company breakdown and architecture limits"):
            rows = []
            for company, counts in bcd["by_company_correct_total"].items():
                rows.append(
                    {
                        "Company": company,
                        **{labels[a]: f"{counts[a][0]}/{counts[a][1]}" for a in labels},
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                "D beats C on DUK, HD and META, but loses on CSCO (21/25 versus 24/25). "
                "Company results are descriptive, not separate confirmatory tests."
            )
            st.write(bcd["input_limitations"])
            st.caption(
                "C and D use the same three ViT views and output width 64, but unequal parameter counts. "
                "The gate is an internal weight, not a causal explanation or calibrated confidence."
            )
        st.download_button(
            "Download audited B/C/D summary",
            json.dumps(bcd, indent=2),
            file_name="FinIntel_BCD_Verified_Results.json",
            mime="application/json",
        )
        with st.expander("B/C/D source fingerprint"):
            st.code(
                f"{bcd['source_archive_filename']}\nSHA-256: {bcd['archive_sha256']}"
            )
    with audit_tab:
        st.subheader("Every option order")
        table = []
        for seed in [11, 22, 33, 44, 55]:
            row = {"Option-order seed": seed}
            for arm in ["native", "model224"]:
                r = next(
                    x
                    for x in evidence["qwen"]["per_seed"]
                    if x["seed"] == seed and x["split"] == split and x["arm"] == arm
                )
                row[ARM_NAMES[arm]] = (
                    f"{r['correct']}/{r['total']} · {100 * r['accuracy']:.2f}%"
                )
            table.append(row)
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption(
            "All groups complete; missing = 0. Original-image Train has two invalid responses per order, counted incorrect. All other groups have zero invalid responses."
        )
        st.subheader("Option-order stability")
        stability_rows = []
        for arm in ["native", "model224"]:
            s = stability(evidence["logs"], split, arm)
            stability_rows.append(
                {
                    "Image": ARM_NAMES[arm],
                    "Same valid option in all 5": f"{s['stable_valid']}/{s['total']}",
                    "Correct in all 5": f"{s['correct_all_five']}/{s['total']}",
                    "Invalid in at least 1": s["invalid_any"],
                }
            )
        st.dataframe(pd.DataFrame(stability_rows), hide_index=True, width="stretch")
        st.caption(
            "Stability compares mapped option identity, not A/B/C/D letters. It is a descriptive post-run calculation; stability does not imply correctness."
        )
        st.subheader("Verification boundary")
        for boundary in meta["boundaries"]:
            st.write("• " + boundary)
        st.write(
            f"Model: {meta['model']} · frozen NF4 · greedy decoding · four original options"
        )
        st.download_button(
            "Download checked score summary",
            json.dumps(evidence["qwen"], indent=2),
            file_name="FinIntel_Qwen_Five_Order_Saved_Log_Summary.json",
            mime="application/json",
        )
        with st.expander("Source fingerprints"):
            st.code(
                f"Source branch commit: {meta['source_commit']}\nSaved notebook SHA-256: {meta['notebook_sha256']}\nQwen model revision: {meta['revision']}"
            )
    st.divider()
    st.subheader("Original extraction pilot: four training histories")
    st.caption("537 Train / 96 eligible Validation questions · one seed (20261031) · five epochs. These older runs are separate from the newer narrative B/C/D comparison above.")
    st.info("The local-context text pilot uses cell, row and header text. It is not the planned narrative-only Model A. The adaptive visual gate is not the full proposed Adaptive Reading method.")
    history = pd.read_csv(ROOT / "data/meeting06/original_pilot_epoch_data.csv")
    history_labels = {"text_only": "Local-context text", "full_table_text_b": "Full-table text", "baseline": "Concatenation", "adaptive_pilot": "Adaptive visual gate"}
    chosen = st.selectbox("Explore an original pilot training history", list(history_labels), format_func=history_labels.get, key="m06_original_history")
    model_history = history.loc[history["model"] == chosen]
    curve_rows = []
    for row in model_history.to_dict("records"):
        for partition in ["train", "validation"]:
            curve_rows.append({"Epoch": row["epoch"], "Accuracy (%)": row[f"{partition}_accuracy_percent"], "Partition": partition.title()})
    st.altair_chart(alt.Chart(pd.DataFrame(curve_rows)).mark_line(point=True).encode(x=alt.X("Epoch:O"), y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0,100])), color="Partition:N", tooltip=["Epoch", "Partition", alt.Tooltip("Accuracy (%):Q", format=".2f")]), width="stretch")
    selected_rows = history.loc[history["selected_checkpoint"].astype(str).str.lower() == "true"].copy()
    st.dataframe(pd.DataFrame([{"Pilot": history_labels[row["model"]], "Selected epoch": int(row["epoch"]), "Validation correct": f"{int(row['validation_correct_fixed_post_epoch'])}/96", "Validation accuracy (%)": round(row["validation_accuracy_percent"], 2)} for row in selected_rows.to_dict("records")]), hide_index=True, width="stretch")
    st.caption("Checkpoints were selected using reused Validation accuracy, not an independent Test set. Different inputs and model capacities prevent attributing the differences to fusion alone. These are verified saved training histories, not newly replayed training.")
    with st.expander("View all four original pilot curves"):
        st.image(str(ROOT / "assets/meeting06/original_four_training_histories.png"), caption="Original pilot: Train and Validation accuracy across epochs", width="stretch")
    st.download_button("Download original pilot epoch data", (ROOT / "data/meeting06/original_pilot_epoch_data.csv").read_bytes(), file_name="original_pilot_epoch_data.csv", mime="text/csv")
    with st.expander("Original concatenation pilot: saved loss and accuracy chart"):

        st.write("Original concatenation pilot, 537 Train and 96 eligible Validation questions, five epochs. This is not the newer narrative B/C/D comparison.")
        st.image(str(ROOT / "assets/meeting06/original_extraction_curves.png"), caption="Original saved notebook chart: loss and exact-cell accuracy", width="stretch")
        st.caption("Selected checkpoint: epoch 3, 59/96 Validation correct (61.46%). Reused development data, not an independent Test result.")
    st.caption(
        "Read-only saved-results review. No live model calls, training, external uploads, publishing, or changes to the running Colab."
    )
