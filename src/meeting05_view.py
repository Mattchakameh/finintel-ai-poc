"""Meeting 05 presentation only: no training, remote writes or private artifact copies."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.meeting05_five_seed import render_five_seed
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
    st.caption('Current source-assisted evaluation, with earlier experiments kept separately for reference.')
    overview, results, methods = st.tabs([
        'Research Overview', 'Cohort-Specific Results', 'Methods, Requirements & Evidence',
    ])
    with overview:
        render_five_seed()
        st.subheader('What is done — and what remains')
        st.markdown('**Done:** five-seed comparison and source checks; mean ± SD reporting; documented experiments and limitations.\n\n**Still open:** improvement in the learned model itself, evaluation on genuinely new examples, and a new company-aware power calculation. The approximately 816-target figure is conditional historical planning.')
        st.caption('New-data inventory: 225 compatible facts; 0 benchmark-ready examples. This is not a claim that the data is ready for training.')
        render_resources()
        with st.expander('Historical A–D comparison (Meeting4) and earlier diagnostics'):
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
        st.info('Additional and historical evaluations use different examples or methods. Start with Research Overview for the current 114-example comparison.')
        with st.expander('Explore separate cohorts and earlier experiments'):
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
    with methods:
        with st.expander('Technical model inputs and gate diagnostics'):
            render_revision(revision, 'models')
            st.dataframe(pd.DataFrame(summary['models']), hide_index=True, use_container_width=True)
            st.info('A/B/C/D are all neural configurations. B is text plus table, not image-only. Original D uses sigmoid gated fusion, not cross-attention. Five seeds are repeated runs, not five architectures.')
            st.write('The Meeting 05 CandidateTileHead has its own implementation and fresh CONTROL/WRONG_ROW heads. It does not load original Model D weights. Whole-image T=1 attention has one valid token, so attention weight is necessarily 1 and does not prove learned spatial selection.')
            st.caption('Source-verified against original training classes and M05 candidate_tile_head.py / training_driver.py. Earlier meeting pages remain historical snapshots.')
        st.divider()
        render_revision(revision, 'tracker')
        st.divider()


def render_resources():
    private_resources = os.environ.get('M05_PRIVATE_RESOURCES')
    if private_resources:
        resources = json.loads(Path(private_resources).read_text())
        st.subheader('PowerPoint, Final Report & Notebook')
        for resource in resources['resources']:
            if not resource.get('visible', True):
                continue
            st.markdown(f"[{resource['name']}]({resource['url']})")
            st.caption(resource['status'])
        st.caption('Verified private resources; existing access permissions apply.')
    else:
        st.subheader('PowerPoint, Final Report & Notebook')
        st.markdown('**PowerPoint — 51 slides** · Private access\n\n**Final Report** · Private access\n\n**Notebook** · Private access')
        st.info('These resources remain private. Public links and downloads are not included here.')
    st.caption('Notebook execution behavior depends on its version. Current private notebook resources include historical material; Colab runtime has not been tested. Saved research results here are displayed without running models.')
