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
    if 'final_robustness_audit' in data:
        validate_robustness(data['final_robustness_audit'])
    if 'post_M5_CONTROL_source_hybrid_v13' in data:
        validate_control(data['post_M5_CONTROL_source_hybrid_v13'])
    return data


def is_pilot(cohort):
    return cohort.startswith(('later_Pilot', 'post_M5'))


def metric_table(metrics):
    return [{'Model / condition': m.get('display_name', m['model']), 'Metric': m['metric'], 'Mean': m['mean'],
             'Seed SD': m['sd'], 'Units': m['units'], 'Targets': m['targets'],
             'Seeds': m['seeds'], 'Label version': ('source_verified_challenge' if m['label_version'] == 'dot_challenge_source_verified' else m['label_version']),
             'Historical cell (1-based)': m.get('historical_cell_1_based'),
             'Evidence source': ('Meeting 05 requirement tracker' if m.get('source_reference') == 'M05_dot_22_item_tracker.json' else m.get('source_reference'))} for m in metrics]


def render_revision(data, section):
    import pandas as pd
    import streamlit as st
    if section == 'overview':
        if 'post_M5_CONTROL_source_hybrid_v13' in data:
            render_control(data['post_M5_CONTROL_source_hybrid_v13'])
        if 'final_robustness_audit' in data:
            st.subheader('Final robustness correction — exploratory pilot')
            st.warning(data['final_robustness_audit']['conclusion'])
            st.caption(data['final_robustness_audit']['model'])
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
        st.caption('Complete means the scoped experiment or reporting requirement is fulfilled, including negative results. Partial and Open identify remaining work; completion does not imply scientific success.')
        st.subheader('Experiment and reporting requirements')
        st.dataframe(pd.DataFrame([{'Requirement': r['requirement'], 'Current status': {'MET': 'Complete', 'PARTIAL': 'Partial', 'OPEN': 'Open'}[r['status']], 'Evidence and limits': r['remaining_scope']} for r in data['advisor_atom_crosswalk'] if r['id'] not in {'R20', 'R21'}]), hide_index=True)
        st.caption('Current status describes the scoped requirement. Completed reporting does not establish scientific improvement; historical input-audit ratings are retained in the underlying evidence.')
        with st.expander('Grouped research status summary'):
            st.dataframe(pd.DataFrame([{'Requirement': r['id'].replace('_', ' '), 'Current status': {'MET': 'Complete', 'PARTIAL': 'Partial', 'OPEN': 'Open'}[r['status']], 'Evidence and limits': r['scope']} for r in data['requirements'] if r['id'] != 'literature_review']), hide_index=True)
        render_power_planning(data)


def render_historical(data):
    import pandas as pd
    import streamlit as st
    evidence = data['historical_temporal_interpretation']
    st.subheader('Historical temporal selector and locked fusion — distinct archived results')
    st.caption(evidence['scope'])
    st.write(evidence['summary'])
    st.info(evidence['R5_scope'])
    with st.expander('Historical saved observations and evidence sources'):
        render_audit_details(evidence['stored_observations'])
        st.dataframe(pd.DataFrame(evidence['sources']), hide_index=True)
        st.warning('Not established: ' + '; '.join(evidence['not_established']))


def render_decoder(data):
    import pandas as pd
    import streamlit as st
    diagnostic = data['post_M5_numeric_decoder_diagnostic']
    if 'final_robustness_audit' in data:
        render_robustness(data['final_robustness_audit'])
        st.subheader('Earlier exploratory decoder history — unequal coverage')
        st.warning('The earlier uniform-versus-learned comparison below is superseded for attribution by the matched-coverage correction above. It is retained as exploratory history, not evidence of learned grounding advantage.')
    else:
        st.subheader('Exploratory numeric decoder — system change, not neural improvement')
    st.caption('Cohort: ' + diagnostic['cohort'] + '; metric: ' + diagnostic['metric'])
    st.warning(diagnostic['summary'])
    st.caption(diagnostic['sd_definition'])
    labels = {'neural': 'Neural baseline', 'uniform': 'Uniform numeric mass',
              'learned': 'Learned numeric mass', 'shuffled_weights': 'One-position cyclic-shift mass',
              'zero_weights': 'Zero mass / neural fallback'}
    rows = [{'Method': r['arm'], 'Decoder': labels[r['decoder']],
             'Mean answer credit (%)': r['mean'], 'Seed sample SD (pp)': r['sample_sd'],
             'Numeric decoder accepted / 115': r['accepted'], 'Neural fallback / 115': r['abstained']}
            for r in diagnostic['results']]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info(diagnostic['control_definition'])
    st.warning(diagnostic['provenance_limit'])
    st.caption(diagnostic['protocol_timing'])
    with st.expander('Decoder verification'):
        st.write(diagnostic['oracle_scope'])
        render_audit_details(diagnostic['verification'])


def validate_robustness(evidence):
    expected_columns = ['arm', 'control', 'strict_mean_pct', 'strict_seed_sd_pp', 'tie_credit_mean_pct']
    if evidence['metric_columns'] != expected_columns:
        raise ValueError('Robustness metric definitions changed')
    if (evidence['common_acceptance'], evidence['common_fallback']) != (106, 9):
        raise ValueError('Robustness coverage must be matched')
    expected = {(arm, control) for arm in ('no_aux', 'row_column_cell_aux')
                for control in ('matched_uniform', 'matched_learned', 'matched_random')}
    rows = evidence['results']
    if len(rows) != 6 or {(r[0], r[1]) for r in rows} != expected:
        raise ValueError('Missing matched robustness control')
    for row in rows:
        if len(row) != 5 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row[2:]):
            raise ValueError('Invalid robustness metric')
        if not (0 <= row[2] <= 100 and row[3] >= 0 and 0 <= row[4] <= 100):
            raise ValueError('Robustness metric outside range')
    for key in ('model', 'cohort', 'random_control', 'strict_ties', 'independence',
                'conclusion', 'scope_limit', 'source_hashes'):
        if not evidence[key]:
            raise ValueError('Missing robustness context')
    if evidence['originalD_modified_or_evaluated']:
        raise ValueError('Robustness must remain distinct from original D')


def render_robustness(evidence):
    import pandas as pd
    import streamlit as st
    validate_robustness(evidence)
    st.subheader('Correction: matched-coverage numeric decoder controls')
    st.warning(evidence['conclusion'])
    st.caption('Model: ' + evidence['model'] + '. Cohort: ' + evidence['cohort'])
    st.info('Every method/control: 106 accepted and 9 neural fallbacks out of 115 targets. '
            'Strict accuracy and fractional tie credit are separate metrics.')
    frame = pd.DataFrame(evidence['results'], columns=evidence['metric_columns']).rename(columns={
        'arm': 'Method', 'control': 'Matched control', 'strict_mean_pct': 'Strict accuracy mean (%)',
        'strict_seed_sd_pp': 'Strict seed sample SD (pp)', 'tie_credit_mean_pct': 'Tie-credit mean (%)'})
    st.dataframe(frame, hide_index=True)
    st.caption(evidence['random_control'])
    st.caption('Strict tie rule: ' + evidence['strict_ties'])
    st.warning(evidence['independence'])
    st.warning(evidence['scope_limit'])
    with st.expander('Robustness verification'):
        render_audit_details(evidence['order_checks'])
        render_audit_details(evidence['feasibility'])


def validate_control(evidence):
    if evidence['seeds'] != [11, 22, 33, 44, 55]:
        raise ValueError('CONTROL requires five fixed seeds')
    if evidence['certified_counts']['Validation'] != 114 or evidence['all_counts']['Validation'] != 115:
        raise ValueError('CONTROL cohort mismatch')
    if 'NOT original Model D' not in evidence['model'] or 'code_proof' in evidence:
        raise ValueError('CONTROL identity or public scope mismatch')
    rows = evidence['validation_results']
    if len(rows) != 5 or {r['arm'] for r in rows} != {'baseline', 'number_only', 'row_period', 'strict_scope_unit', 'random_override'}:
        raise ValueError('Missing CONTROL arm')
    for row in rows:
        for key in ('certified_Val114_mean_sd_percent', 'all_Val115_mean_sd_percent'):
            mean, sd = row[key]
            if not all(math.isfinite(x) for x in (mean, sd)) or not 0 <= mean <= 100 or sd < 0:
                raise ValueError('Invalid CONTROL metric')
    if evidence['order_copy']['new_accuracy_measured'] is not False:
        raise ValueError('Order checks are not accuracy')


def render_control(evidence):
    import pandas as pd
    import streamlit as st
    st.subheader('CandidateTileHead CONTROL SYSTEM — posthoc readout')
    st.caption(evidence['model'])
    st.warning('Reused development Validation, four companies. Extra full HTML and selected-cell unit metadata change the available information. No neural-learning, joint-accuracy, H1/H2, fresh-Test or independent-generalization claim.')
    st.caption('Value-only results; mean ± seed SD across five fixed seeds: 11, 22, 33, 44, 55. Posthoc system comparison.')
    labels = {'baseline': 'Baseline', 'number_only': 'Number only', 'row_period': 'Row + period',
              'strict_scope_unit': 'Strict scope + unit', 'random_override': 'Random override'}
    def table(key):
        return pd.DataFrame([{'System method': labels[r['arm']], 'Value accuracy (%)': r[key][0],
                              'Seed SD (pp)': r[key][1]} for r in evidence['validation_results']])
    st.markdown('**Certified reused Validation — 114 targets**')
    st.dataframe(table('certified_Val114_mean_sd_percent'), hide_index=True)
    st.info('Row + period (73.3333%) exceeds strict scope + unit (67.8947%) on certified Validation 114. A strict-system advantage is not established.')
    with st.expander('Separate legacy cohort — all Validation 115 targets'):
        st.caption('Different denominator and certification scope; do not combine with Validation 114 or original frozen Test 130.')
        st.dataframe(table('all_Val115_mean_sd_percent'), hide_index=True)
    for limit in evidence['limits']:
        st.caption(limit)
    st.warning('Five proposed value/candidate patches remain unapplied; 11 quarantines remain. The data package is not fully cleaned or training-ready.')
    with st.expander('Company coverage and data/order audit — counts are not accuracy'):
        st.write('Strict overrides by company (counts, not company accuracy):')
        render_audit_details(evidence['company_strict_overrides'])
        st.write('Reference status counts (not accuracy):')
        render_audit_details(evidence['reference_status_counts'])
        st.write('Reversible order-copy checks; no new accuracy measured:')
        render_audit_details(evidence['order_copy'])
        st.caption('The 60 forward pairs cover 12 unique targets; repeated route/deletion checks are not independent evaluation samples.')


def render_power_planning(data):
    import pandas as pd
    import streamlit as st
    power = data['power']
    st.subheader('Conditional power planning')
    st.write(f"**Approximately {power['required_n']:,} independent targets** under the archived assumptions. This is conditional planning, not a completed power recalculation or a confirmed sample-size recommendation.")
    st.dataframe(pd.DataFrame([
        ('Statistical test', power['statistical_reproduction']),
        ('Significance level (alpha)', f"{power['alpha']:.2f}"),
        ('Target power', f"{power['power']:.0%}"),
        ('Archived sample', f"{power['targets']} targets across {power['companies']} companies"),
        ('Analysis unit', 'Target-level paired difference, averaged across seeds'),
        ('Independence assumption', 'Targets assumed independent; company clustering is not adjusted'),
    ], columns=['Planning assumption', 'Description']), hide_index=True)
    st.warning(power['caveat'])
    with st.expander('Technical details: archived power calculation'):
        st.dataframe(pd.DataFrame([
            ('Mean paired difference (percentage points)', f"{power['mean_difference_pp']:.4f}"),
            ('Paired SD (percentage points)', f"{power['paired_sd_pp']:.4f}"),
            ('Variance (squared percentage points)', f"{power['variance_pp_squared']:.4f}"),
            ('Standardized paired effect (dz)', f"{power['dz']:.6f}"),
            ('Estimated power in the archived sample', f"{power['current_power']:.2%}"),
            ('Historical notebook cell (1-based)', str(power['historical_cell_1_based'])),
            ('Original implementation verified', 'Yes' if power['original_implementation_verified'] else 'No — original implementation source was unavailable'),
        ], columns=['Archived quantity', 'Value']), hide_index=True)


def render_audit_details(value):
    """Readable retained audit facts, without serialized objects or source digests."""
    import streamlit as st
    if isinstance(value, dict):
        for key, item in value.items():
            if any(token in key.lower() for token in ('sha256', 'hash')):
                continue
            label = key.replace('_', ' ').capitalize()
            if isinstance(item, (dict, list)):
                st.markdown(f'**{label}**')
                render_audit_details(item)
            else:
                shown = 'Not established' if item is None else ('Yes' if item is True else 'No' if item is False else str(item))
                st.write(f'{label}: {shown}')
    elif isinstance(value, list):
        for item in value:
            render_audit_details(item)
    else:
        st.write(str(value))
