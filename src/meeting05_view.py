"""Meeting 05 presentation only: no training, remote writes or private artifact copies."""
import pandas as pd
import streamlit as st

from src.meeting05_loader import Meeting05EvidenceError, load_meeting05
from src.meeting05_revision import load_revision, render_revision


def render_meeting05():
    try:
        summary = load_meeting05()
        revision = load_revision()
    except (Meeting05EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        st.info('Restore the reviewed data/meeting05/summary.json. No substitute results are shown.')
        return

    st.title('FinIntel AI — Meeting 05')
    st.subheader('Financial Grounding: Audit, Repairs & Controlled Diagnostics')
    st.caption('Archived research evidence as of ' + summary['as_of'] + '. This dashboard does not train models or change the source notebooks.')
    st.warning('No demonstrated general neural accuracy improvement. Hybrid rules and structured-source extraction are different information regimes, not neural gains.')
    overview, results, models, tracker, evidence = st.tabs([
        'Research Overview', 'Cohort-Specific Results', 'Models & Inputs',
        'Requirement Evidence', 'Provenance & Artifacts',
    ])
    with overview:
        render_revision(revision, 'overview')
        st.markdown('### What improved — and what did not')
        st.markdown('''
- **Evidence reliability:** source alignment, fiscal references, period provenance and table selection were audited and repaired in separate copies.
- **Corrected 115-target replay:** frozen neural **41.04%**; numeric-rule + neural fallback **53.57%**. Exploratory hybrid benefit, not new neural weights.
- **New challenges remain difficult:** annual/quarter **19.00%** on 20 queries; similar-concept text head **39.25% → 37.25%** on 80 queries after input repair.
- **Structured source extraction:** **104/105** correct after HTML/XBRL cell/header repair on the eligible cohort. This adds source information and is **not 99% neural performance**.
''')
        st.info('Financial grounding means linking a value to the correct company, concept, period, unit and source evidence — not just recognizing a number.')
        st.markdown('### Research boundary')
        st.write('No new Test evaluation. Reused Validation, answer-assisted historical evidence windows, small correlated challenges and provided source catalogs limit generalization. Completion of scoped deliverables does not confirm the hypotheses.')
        st.caption('Simple year matching (104/115) and temporal grounding (105/115) used candidate-period metadata on historical Validation; neither is deployment extraction accuracy.')
    with results:
        render_revision(revision, 'results')
        st.info('Select one matched cohort at a time. These results must not be pooled into a single leaderboard.')
        selected = st.selectbox('Evidence cohort', [r['id'] for r in summary['results']],
                                format_func=lambda value: next(r['title'] for r in summary['results'] if r['id'] == value), key='m05_cohort')
        result = next(r for r in summary['results'] if r['id'] == selected)
        st.subheader(result['title'])
        st.markdown('**Cohort:** ' + result['cohort'])
        st.markdown('**Information regime:** ' + result['regime'])
        frame = pd.DataFrame(result['rows'])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.bar_chart(frame.set_index('Method'))
        st.warning(result['limit'])
    with models:
        render_revision(revision, 'models')
        st.dataframe(pd.DataFrame(summary['models']), hide_index=True, use_container_width=True)
        st.info('A/B/C/D are all neural configurations. B is text plus table, not image-only. Original D uses sigmoid gated fusion, not cross-attention. Five seeds are repeated runs, not five architectures.')
        st.write('The Meeting 05 CandidateTileHead has its own implementation and fresh CONTROL/WRONG_ROW heads. It does not load original Model D weights. Whole-image T=1 attention has one valid token, so attention weight is necessarily 1 and does not prove learned spatial selection.')
        st.caption('Source-verified against original training classes and M05 candidate_tile_head.py / training_driver.py. Earlier meeting pages remain historical snapshots.')
    with tracker:
        render_revision(revision, 'tracker')
    with evidence:
        render_revision(revision, 'provenance')
        st.warning('Linked Drive artifacts retain their existing owner-only / authorized access. This app does not grant access, copy private notebooks or broaden sharing. Artifact revision and publication are coordinated separately; no deck/page count is asserted here.')
        for title, url in summary['links'].items():
            st.markdown(f'[{title}]({url})')
        st.caption('Safe notebook: historical executable cells are archived; visible history is non-executable. Run All only inspects the review archive, not the original training workflow.')
        with st.expander('Snapshot provenance and SHA-256 digests'):
            st.dataframe(pd.DataFrame(summary['provenance']), hide_index=True, use_container_width=True)
        st.caption('Compact summaries only. Full private source artifacts are not bundled. No reported performance is recomputed by this display layer.')
