# Local validation — 2026-10-24

Base commit: `b834122316c5f1102301eea82285cde7cd93a509`.
Branch: `meeting05-dashboard-local` in isolated `finintel_m05_dashboard` checkout.
Branch-only publication to a new `meeting05-dashboard` is authorized.
No pull request, merge or live deployment performed.

## Passed

- `python -m pytest tests/test_meeting05.py -q`: **10 passed** in 3.52 seconds after final deck-status update.
- AppTest used actual archived POC checkpoints (torch/FAISS), not stub data.
- Correct initial Meeting 02 view; repeated 02→03→04→05→02→05→03→02→04→05
  switching renders one correct title and no exceptions.
- All six Meeting 05 cohort selections render with cohort context.
- Missing Meeting 05 summary gives explicit error without substitutes and permits
  navigation away; missing Meeting 04 visual directory displays existing warning.
- Invalid/missing summary JSON and out-of-range metric rejected.
- Scientific invariant checks retain negative neural result, corrected115 comparison,
  source105 denominator, parser issuer limitation and LR exclusion.
- `ruff check app.py src/meeting05_loader.py src/meeting05_view.py tests/test_meeting05.py`: passed.
- `python -m compileall -q app.py src tests`: passed.
- `git diff --check`: passed.
- Original M02 body after Data loading marker is byte-identical.
- Original M03 evaluation block is AST-identical after indentation removal.
- M04 source/data and all historical assets are unchanged.

Runtime used: Streamlit 1.66.1, torch 2.14.1+cpu, pandas 3.1.0,
numpy 2.5.4, faiss 1.15.1. Isolated venv: `/tmp/m05_dashboard_venv`.
The initial global pip attempt failed read-only; recovered via writable venv.
Initial AppTest path setup was corrected to absolute path before the final pass.
Ruff import-order findings were fixed and checks rerun.

## Warnings and untested boundaries

18 warnings originate from Streamlit chart internals using pandas deprecated
`reset_index(inplace=True)`. They are nonfatal; no application warnings were hidden.
AppTest checks element trees, not browser pixel/layout, deployment or credentials.
Local Chromium/Playwright visual QA was attempted but browser startup failed:
`process_singleton_posix.cc socket() failed: Operation not permitted`.
No visual browser pass is claimed. Live-cloud visual QA, authenticated Streamlit management, final CI,
and custom-query Hugging Face download were **not tested**. Existing custom-query
implementation is unchanged. Test success does not establish scientific validity
beyond the documented evidence or eliminate research limitations.

## Hosting publication hold

The repository is PUBLIC. Source artifacts have not been copied; links retain
existing owner-only / authorized access. Review these links and compact research
summaries before publication. The supplied live URL maps to the M04 branch; a
separate M05 deployment/branch target needs confirmation. Existing deployments
must remain untouched. Revised 40-slide PowerPoint and 40-page PDF were verified saved by the deck
owner. Links retain their original IDs and access permissions; files are not bundled.
