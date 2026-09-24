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

from src.data_loader import (
    POCDataLoadError,
    compute_retrieval_metrics,
    load_poc_data,
    run_pipeline_validation,
)
from src.retrieval import build_faiss_index, live_search
from src.meeting03_loader import (
    Meeting03EvidenceError,
    load_and_validate_meeting03,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="FinIntel AI -- POC Demo",
    layout="wide",
)
# ============================================================
# Meeting navigation
# ============================================================

dashboard_view = st.sidebar.radio(
    "FinIntel AI Research Dashboard",
    [
        "Meeting 02 — Pipeline Feasibility",
        "Meeting 03 — Learning & Evaluation",
    ],
    index=0,
)

st.sidebar.caption(
    "Meeting 02 establishes technical pipeline feasibility. "
    "Meeting 03 extends the system to supervised learning "
    "and temporal held-out evaluation."
)
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
