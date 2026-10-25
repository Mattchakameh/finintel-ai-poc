"""Aggregate-only original D replay; distinct from historical Test and CONTROL."""
import json
import math
import statistics
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'data/meeting05/original_d_five_seed.json'


def load_five_seed():
    data = json.loads(PATH.read_text())
    counts = {'all115_legacy': 115, 'certified114_system_yield': 114,
              'certified_candidate_available113': 113}
    for row in data['results']:
        if row['N_per_seed'] != counts[row['cohort']] or row['model_seeds'] != [11, 22, 33, 44, 55]:
            raise ValueError('Original D replay cohort or seed mismatch')
        for key in ('value_accuracy_percent', 'concept_accuracy_percent', 'joint_accuracy_percent'):
            metric = row[key]
            values = list(metric['per_model_seed'].values())
            if len(values) != 5 or not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
                raise ValueError('Invalid five-seed metric')
            if not math.isclose(statistics.mean(values), metric['mean'], abs_tol=1e-8):
                raise ValueError('Five-seed mean mismatch')
            if not math.isclose(statistics.stdev(values), metric['sample_sd'], abs_tol=1e-8):
                raise ValueError('Five-seed SD mismatch')
    return data


def render_five_seed():
    import pandas as pd
    import streamlit as st
    data = load_five_seed()
    st.subheader('Latest original D replay — five fixed seeds, reused Validation')
    st.info('Original D core is unchanged sigmoid gated fusion. Source-assisted wrappers add recovered full HTML and selected-cell unit metadata; these are system gains, not neural learning or generalization.')
    st.markdown('**Original D → added source check → source-reference 114 results**')
    st.write('Baseline: **28.42% ± 3.08** → row/period source check: **62.81% ± 2.60**.')
    st.caption('Value agreement, mean ± sample SD across five fixed seeds. Same reused source-reference 114 targets; additional source information, not improved neural weights.')
    headline_rows = {r['arm']: r for r in data['results']
                     if r['cohort'] == 'certified114_system_yield' and r['variant'] == 'metadata_assisted'}
    compact = []
    for arm, label, meaning in [('baseline', 'Original D', 'Model prediction only'),
                                ('row_period', 'Original D + source check', 'Checks the source row and period')]:
        metric = headline_rows[arm]['value_accuracy_percent']
        compact.append({'Comparison': label, 'What changes': meaning,
                        'Value agreement — mean ± seed SD': f"{metric['mean']:.2f}% ± {metric['sample_sd']:.2f} pp"})
    st.dataframe(pd.DataFrame(compact), hide_index=True)

    with st.expander('Technical results — strict method, separate cohorts and concept/joint metrics'):
        labels = {'certified114_system_yield': 'Source-reference 114 — system yield',
                  'all115_legacy': 'Legacy 115 — original-label agreement',
                  'certified_candidate_available113': 'Candidate-available 113 — restricted subset'}
        cohort = st.selectbox('Original D replay cohort', list(labels), format_func=labels.get, key='m05_d_replay_cohort')
        variant = st.selectbox('Original D wrapper information', ['metadata_assisted', 'visible_only'], key='m05_d_replay_variant')
        rows = [r for r in data['results'] if r['cohort'] == cohort and r['variant'] == variant]
        table = []
        for row in rows:
            item = {'Method': row['arm'], 'Targets per seed': row['N_per_seed']}
            for key, label in [('value_accuracy_percent', 'Value'), ('concept_accuracy_percent', 'Concept'), ('joint_accuracy_percent', 'Joint')]:
                item[label + ' mean (%)'] = row[key]['mean']
                item[label + ' seed SD (pp)'] = row[key]['sample_sd']
            table.append(item)
        st.dataframe(pd.DataFrame(table), hide_index=True)
    st.caption('Mean ± sample SD across seeds 11, 22, 33, 44, 55. Random controls average five randomizations within each model seed. SD is not a target/cluster confidence interval.')
    st.warning('Row/period exceeds strict. Five seeds share reused Validation targets in four companies, not 575 independent examples. No new Test evaluation, neural-learning, H1/H2 or generalization conclusion.')
    st.caption('Source-reference 114 includes one reference absent from the original options; candidate-available 113 excludes that case. Legacy 115 measures original-label agreement, not source truth. Original options remain unchanged.')
    st.caption('Visible-only wrappers make no overrides. Concept logits remain unchanged. Historical Test130 and CandidateTileHead CONTROL use different conditions and remain separate below.')
    st.info('New-data inventory: 225 compatible facts; 0 benchmark-ready examples. Counts are not accuracy or evidence of training readiness.')
    with st.expander('Replay verification and aggregate source hashes'):
        st.write('Independent saved-output audit accepted all 575 baseline choices and 10,350 wrapper rows. Choice parity does not establish bitwise-logit or historical-concept parity; the exact historical model revision is unrecorded.')
        st.json({k: data[k] for k in ('source_sha256', 'audit_report_sha256')})
