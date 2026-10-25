"""Public-safe reviewed research summary; no source execution or remote access."""
import json
import math
import statistics
from pathlib import Path

REVISION_PATH = Path(__file__).resolve().parents[1] / 'data/meeting05/revision.json'


def load_revision(path=None):
    data = json.loads(Path(path or REVISION_PATH).read_text())
    if data['schema_version'] != 1:
        raise ValueError('Unsupported revision schema')
    encoded = json.dumps(data)
    for forbidden in ('/Users/', 'drive.google.com', 'libfile_', 'source_file', 'download_url', '@gw'):
        if forbidden in encoded:
            raise ValueError('Private source metadata is not permitted')
    ids = [m['id'] for m in data['metrics']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate metric ID')
    for m in data['metrics']:
        for key in ('cohort', 'label_version', 'metric', 'units', 'caveat', 'source_sha256'):
            if not m.get(key):
                raise ValueError('Missing metric context: ' + key)
        if not isinstance(m['mean'], (float, int)) or not math.isfinite(m['mean']):
            raise ValueError('Invalid metric mean')
        if m['units'] == 'percent' and not 0 <= m['mean'] <= 100:
            raise ValueError('Percentage outside range')
        if m['sd'] is not None and (not math.isfinite(m['sd']) or m['sd'] < 0):
            raise ValueError('Invalid SD')
    if any(r['status'] not in {'MET', 'PARTIAL', 'OPEN'} for r in data['requirements']):
        raise ValueError('Invalid requirement status')
    atoms = data['advisor_atom_crosswalk']
    if [row['id'] for row in atoms] != [f'R{i}' for i in range(1, 22)]:
        raise ValueError('Incomplete requirement crosswalk')
    if any(row['status'] not in {'MET', 'PARTIAL', 'OPEN'} for row in atoms):
        raise ValueError('Invalid requirement status')
    chance = next(m for m in data['metrics'] if m['id'] == 'chance')
    if chance['mean'] != 25 or chance['cohort'] != 'original_frozen_Test_130':
        raise ValueError('Invalid chance reference')
    if {g['seed'] for g in data['gates']} != {11, 22, 33, 44, 55}:
        raise ValueError('Missing original D seed')
    diagnostic = data['post_M5_numeric_decoder_diagnostic']
    if diagnostic['targets'] != 115 or diagnostic['metric'] != 'tie_aware_answer_credit_pct':
        raise ValueError('Invalid decoder cohort or metric')
    if diagnostic['new_training'] or diagnostic['new_Test_evaluation']:
        raise ValueError('Decoder must remain an archived diagnostic')
    pairs = {(r['arm'], r['decoder']) for r in diagnostic['results']}
    expected = {(a, d) for a in ('no_aux', 'row_column_cell_aux')
                for d in ('neural', 'uniform', 'learned', 'shuffled_weights', 'zero_weights')}
    if pairs != expected or len(diagnostic['results']) != 10:
        raise ValueError('Missing paired decoder control')
    for row in diagnostic['results']:
        if row['accepted'] + row['abstained'] != 115 or len(row['seeds']) != 5:
            raise ValueError('Invalid decoder coverage')
        if not math.isclose(statistics.mean(row['seeds']), row['mean'], abs_tol=1e-8):
            raise ValueError('Decoder mean disagrees with seeds')
        if not math.isclose(statistics.stdev(row['seeds']), row['sample_sd'], abs_tol=1e-8):
            raise ValueError('Decoder SD disagrees with seeds')
    return data


def is_pilot(cohort):
    return cohort.startswith(('later_Pilot', 'post_M5'))


def metric_table(metrics):
    return [{'Model / condition': m.get('display_name', m['model']), 'Metric': m['metric'], 'Mean': m['mean'],
             'Seed SD': m['sd'], 'Units': m['units'], 'Targets': m['targets'],
             'Seeds': m['seeds'], 'Label version': m['label_version'],
             'Historical cell (1-based)': m.get('historical_cell_1_based'),
             'Evidence source': m.get('source_reference')} for m in metrics]


def render_revision(data, section):
    import pandas as pd
    import streamlit as st
    if section == 'overview':
        st.subheader('Original A–D results — archived frozen Test, 130 targets')
        st.info('Four-option value-selection chance: 25%. This is not a joint-accuracy significance test.')
        rows = [m for m in data['metrics'] if m['id'].startswith('historical_') and m['metric'] in {'value_accuracy', 'joint_accuracy'}]
        st.dataframe(pd.DataFrame(metric_table(rows)), hide_index=True)
        st.caption(rows[0]['caveat'])
        h1 = next(m for m in data['metrics'] if m['id'] == 'H1_C_minus_B')
        st.warning(h1['caveat'])
        st.caption(data['cohort_policy'])
        render_historical(data)
    elif section == 'results':
        st.subheader('Audited diagnostics — separate cohorts')
        group = st.radio('Evidence period', ['Meeting 05 and historical evidence', 'Later Pilot — exploratory appendix'], key='m05_period')
        selected_metrics = [m for m in data['metrics'] if is_pilot(m['cohort']) == (group == 'Later Pilot — exploratory appendix')]
        cohorts = list(dict.fromkeys(m['cohort'] for m in selected_metrics))
        cohort = st.selectbox('Audited evidence cohort', cohorts, key='m05_audited_cohort')
        rows = [m for m in data['metrics'] if m['cohort'] == cohort]
        st.write('Cohort: ' + cohort)
        st.dataframe(pd.DataFrame(metric_table(rows)), hide_index=True)
        for caveat in dict.fromkeys(m['caveat'] for m in rows):
            st.warning(caveat)
        for definition in dict.fromkeys(m['sd_definition'] for m in rows if m.get('sd_definition')):
            st.caption(definition)
        if group == 'Later Pilot — exploratory appendix':
            render_decoder(data)
    elif section == 'models':
        st.subheader('Original model input contract')
        st.dataframe(pd.DataFrame(data['model_inputs']), hide_index=True)
        vision = next(m for m in data['metrics'] if m['id'] == 'financial_evidence_vision_baseline')
        st.info(vision['caveat'])
        st.subheader('Original D gates by seed — Validation 115 / 460 candidate rows')
        st.dataframe(pd.DataFrame(data['gates']), hide_index=True)
        st.warning('Gate weights are not causal modality importance; these are original D diagnostics, not CandidateTileHead weights.')
    elif section == 'tracker':
        st.caption(data['requirements_note'])
        st.subheader('All 21 requirement items')
        st.dataframe(pd.DataFrame(data['advisor_atom_crosswalk']), hide_index=True)
        with st.expander('Grouped research status summary'):
            st.dataframe(pd.DataFrame(data['requirements']), hide_index=True)
        st.subheader('Literature review progress — completion remains open')
        st.json(data['literature_review_progress'])
        st.subheader('Conditional power planning')
        st.json({k: v for k, v in data['power'].items() if not k.endswith('sha256')})
        st.warning(data['power']['caveat'])
    elif section == 'provenance':
        st.caption('Canonical review SHA-256: ' + data['canonical_sha256'])
        st.caption('Public manifest SHA-256: ' + data['public_manifest_sha256'])
        st.caption('Historical notebook cells use 1-based numbering. Source references identify evidence without exposing private files.')


def render_historical(data):
    import pandas as pd
    import streamlit as st
    evidence = data['historical_temporal_interpretation']
    st.subheader('Historical temporal selector and locked fusion — distinct archived results')
    st.caption(evidence['scope'])
    st.write(evidence['summary'])
    st.info(evidence['R5_scope'])
    with st.expander('Historical saved observations and evidence sources'):
        st.json(evidence['stored_observations'])
        st.dataframe(pd.DataFrame(evidence['sources']), hide_index=True)
        st.warning('Not established: ' + '; '.join(evidence['not_established']))


def render_decoder(data):
    import pandas as pd
    import streamlit as st
    diagnostic = data['post_M5_numeric_decoder_diagnostic']
    st.subheader('Exploratory numeric decoder — system change, not neural improvement')
    st.caption('Cohort: ' + diagnostic['cohort'] + '; metric: ' + diagnostic['metric'])
    st.warning(diagnostic['summary'])
    st.caption(diagnostic['sd_definition'])
    labels = {'neural': 'Neural baseline', 'uniform': 'Uniform numeric mass',
              'learned': 'Learned numeric mass', 'shuffled_weights': 'One-position cyclic-shift mass',
              'zero_weights': 'Zero mass / neural fallback'}
    rows = [{'Arm': r['arm'], 'Decoder': labels[r['decoder']],
             'Mean answer credit (%)': r['mean'], 'Seed sample SD (pp)': r['sample_sd'],
             'Numeric decoder accepted / 115': r['accepted'], 'Neural fallback / 115': r['abstained']}
            for r in diagnostic['results']]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info(diagnostic['control_definition'])
    st.warning(diagnostic['provenance_limit'])
    st.caption(diagnostic['protocol_timing'])
    with st.expander('Decoder audit and exact source hashes'):
        st.write(diagnostic['oracle_scope'])
        st.write(diagnostic['verification'])
        st.json(diagnostic['source_hashes'])
        st.caption('Audit SHA-256: ' + diagnostic['audit_sha256'])
