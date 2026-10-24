# Meeting 05 — isolated dashboard addition (local review)

Base: `meeting04-dashboard` at `b834122316c5f1102301eea82285cde7cd93a509`.
Working branch: `meeting05-dashboard-local`; authorized review-only remote branch:
`meeting05-dashboard`. **No live deployment or merge authorized.**

## Run and test

Use the existing `requirements.txt`, then:

```sh
streamlit run app.py
python -m pytest tests/test_meeting05.py -q
python -m compileall -q app.py src
```

The Meeting 05 sidebar route loads `src/meeting05_view.py` and the compact
`data/meeting05/summary.json`. It never runs training or edits original evidence.
The JSON records local source artifact SHA-256 hashes for review; original
notebooks, private reports, model weights and raw private records are not copied.
Drive links preserve existing access controls. Links are not public sharing grants.
The repository is public: review this compact data and links before any publication.

Meeting 02/03 routing is repaired only in this new copy: the previously global
Meeting 03 evaluation block is now inside its Meeting 03 condition. Original
Meeting 02 body, Meeting 03 content, Meeting 04 files and existing assets remain
unchanged. Previously the default Meeting 02 route incorrectly showed Meeting 03.

## Scientific interpretation

Do not rank unrelated cohorts together. Original-label 115, corrected-label115,
new period20, new concept80 and source-selected105 are separate experiments.
The corrected frozen neural replay is 41.04%; 53.57% is a numeric-rule/frozen-neural
hybrid, not new neural weights. The 104/105 source result adds HTML/XBRL information,
and the 72/80 parser is conditional on a correct-issuer supplied table. Both are
rules, not neural gains. The 19% period diagnostic and 39.25%→37.25% similar-concept
repair remain negative results. No new Test outcomes were evaluated.

A/B/C/D are neural configurations. B uses text/table, not image-only; D uses a
sigmoid gate, not cross-attention. M05 CandidateTileHead is a separate implementation.
Full details/limitations are visible next to each displayed result. Literature
Review is excluded. The revised 40-slide deck and 40-page PDF are verified saved at their existing
Drive IDs, with unchanged permissions; no full slide files are bundled.

## Review branch and deployment plan

1. Review the new `meeting05-dashboard` branch, compact evidence snapshot and tests.
2. Confirm target new branch and whether Meeting 05 needs its own separate app,
   consistent with historical separate deployments. Current supplied live URL
   was verified to use `meeting04-dashboard/app.py`; do not overwrite it silently.
3. Branch-only publication is authorized. A **draft** PR needs an appropriate
   review target; do not silently target an old deployed branch or `main`.
   Merge and deployment are separate, unapproved actions.
4. Verify remote commit and CI. Provision/repoint a new Meeting 05 deployment only
   with explicit approval, then smoke-test all four routes in live UI.

No authenticated Streamlit Cloud management session is established.
AppTest exercises local rendered element trees. Local Chromium visual QA was
attempted but blocked by socket permissions before browser startup. No browser
visual pass or live-cloud deployment verification is claimed. Live custom retrieval model-download behavior is unchanged.
