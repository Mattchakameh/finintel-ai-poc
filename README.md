# FinIntel AI -- POC Streamlit Demo

A presentation layer over the existing, already-validated FinIntel AI proof
of concept. This app does not retrain, re-derive, or alter any reported
result -- it loads the exact checkpointed outputs of the source research
notebook and displays them.

**POC disclaimer (shown in-app):** This POC demonstrates technical pipeline
feasibility. Results are preliminary and do not establish H1, H2, or H3.

## Folder structure

```
finintel_streamlit/
    app.py                  # Streamlit entry point
    requirements.txt
    README.md
    src/
        __init__.py
        data_loader.py       # Loads + validates the 3 checkpoint files
        retrieval.py          # FAISS rebuild + demo-safe live query search
    data/
        finintel_EOD_2026_09_08_objects.pkl
        finintel_EOD_2026_09_08_status.pkl
        finintel_EOD_2026_09_08_tensors.pt
    assets/
        pair1_aapl-20230930_g2.jpg
        pair2_aapl-20240928_g2.jpg
        pair3_aapl-20250927_g2.jpg
        pair4_goog-20231231_g1.jpg
        pair5_goog-20241231_g1.jpg
        pair6_goog-20251231_g1.jpg
        pair8_nvda-20250126_g2.jpg
```

## Data provenance

- **`data/*.pkl`, `data/*.pt`** -- the source notebook's own "FINAL
  END-OF-DAY CHECKPOINT" (2026-09-08). Independently re-validated before
  being placed here: reloading and re-running the notebook's own 11-check
  pipeline validation against these files reproduces 11/11 passed, and
  recomputing Precision@5 from the stored manual labels reproduces
  21/25 = 0.840 -- both computed live by `app.py`, not hardcoded.
- **`assets/*.jpg`** -- the 7 real SEC financial chart images used by the
  7 clean multimodal pairs (AAPL x3, GOOGL x3, NVDA x1). These were never
  saved as standalone files by the notebook (only ephemeral Colab paths
  were checkpointed), so they were extracted directly from the notebook's
  own rendered output cell ("FINAL FINANCIAL VISUAL INSPECTION"), matched
  to the checkpoint by exact filename + accession number, and cropped
  losslessly from that output. No image was regenerated, edited, or
  substituted.

AMZN and MSFT have zero usable visual pairs in the current POC subset
(confirmed in the checkpoint's own status record) -- they appear in the
15-filing retrieval corpus but not in the 7-pair multimodal demo, and the
app is explicit about this rather than papering over it.

## Installation

```bash
cd finintel_streamlit
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## What's loaded locally vs. downloaded

- **Local, from `data/` and `assets/`:** all 15 filings, 515 chunks, the
  515 MiniLM embeddings, the 25 labeled retrieval results, all 7
  multimodal pairs' FinBERT/ViT/cross-attention/fusion tensors, and the 7
  chart images. None of this requires network access.
- **Downloaded on first run (needs internet):** the MiniLM query encoder
  (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) is downloaded from
  Hugging Face **only** the first time you use the "Custom query (live
  retrieval)" option in the Retrieval tab. It's cached by
  `sentence-transformers` locally after that. The "5 evaluated queries"
  path needs no download at all -- it reads directly from the checkpoint.

## Live-demo safety

The Custom Query path is wrapped so it can never crash the app:
- If the MiniLM model can't be downloaded (no network in the room, a
  firewall, etc.) or the search otherwise fails, the app shows a clear
  error and tells you to switch back to "5 evaluated queries" -- it does
  not stop the app or throw a traceback.
- The 5 evaluated queries and their P@5 figures never depend on network
  access or on anything you type -- they're a fixed replay of the
  manually-judged results, so that part of the demo is not at risk.

**Recommendation:** if presenting somewhere with uncertain wifi, load
the app and run one custom query beforehand so the model is cached
locally before you're in front of your advisor.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "Could not load the POC checkpoint" on startup | One of the 3 files under `data/` is missing or was renamed -- check the filenames above exactly. |
| `faiss` import error | `pip install faiss-cpu` (not `faiss` -- that's a different, GPU-only package name that often fails to install). |
| Custom query hangs or errors | First run downloads the MiniLM model -- needs internet. If it keeps failing, use the 5 evaluated queries instead; nothing else in the app depends on this. |
| Image not showing on Visual Evidence tab | Only pairs 1, 2, 3, 4, 5, 6, 8 have images in `assets/` (pairs 7 and 9 were excluded as decorative/logo images upstream in the POC, not multimodal candidates). |

## Environment

- Python 3.10+
- No GPU required -- everything displayed is precomputed; the only
  runtime model (MiniLM query encoder) runs comfortably on CPU.

## Known limitations

- The multimodal subset is 7 pairs across 3 companies (AAPL, GOOGL,
  NVDA) -- far too small for any performance claim, by design. This is
  stated explicitly in the app's Research Boundary tab.
- The adaptive fusion gate is untrained; modality weights hover near
  50/50 and are labeled in the app as "POC / untrained gate outputs,"
  never as learned importance.
- AMZN and MSFT have MD&A text and retrieval coverage but no chart
  images, so they won't appear in the multimodal-pair selector.

## Live-demo sequence (~2-3 minutes)

1. **Overview tab (15s)** -- point to the disclaimer banner and the
   pipeline diagram; state plainly this is a feasibility demo, not a
   results demo.
2. **Filing & MD&A tab (20s)** -- pick AAPL's most recent filing, show
   the extracted Item 7 text and chunk count.
3. **Retrieval tab (40s)** -- run one of the 5 evaluated queries, show
   the ranked, labeled results and per-query P@5; then switch to Custom
   Query and type a live question to show retrieval works on unseen
   input, explicitly noting it isn't manually scored.
4. **Visual Evidence tab (20s)** -- switch the sidebar pair selector to
   an AAPL or GOOGL pair, show the real SEC chart.
5. **Cross-Attention & Adaptive Fusion tab (30s)** -- show the fused
   representation shape and the modality weights, and read the
   untrained-gate disclaimer aloud.
6. **POC Validation tab (15s)** -- show the live 11/11 check and the
   labeled Precision@5.
7. **Research Boundary tab (15s)** -- close on the "does NOT yet
   demonstrate" list to reinforce the feasibility-only framing.
