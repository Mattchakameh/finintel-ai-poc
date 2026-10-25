"""Public release and cohort guards, runnable without the UI dependencies."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.meeting05_revision import is_pilot, load_revision, metric_table


class RevisionTests(unittest.TestCase):
    def reject(self, data, pattern):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'revision.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, pattern):
                load_revision(path)

    def test_private_source_path_rejected(self):
        data = load_revision()
        data['metrics'][0]['source_file'] = '/Users/private/notebook.ipynb'
        self.reject(data, 'Private')

    def test_invalid_status_rejected(self):
        data = load_revision()
        data['requirements'][0]['status'] = 'ALL_DONE'
        self.reject(data, 'status')

    def test_missing_cohort_rejected(self):
        data = load_revision()
        data['metrics'][0]['cohort'] = ''
        self.reject(data, 'context')

    def test_nonfinite_mean_rejected(self):
        data = load_revision()
        data['metrics'][0]['mean'] = float('nan')
        self.reject(data, 'mean')

    def test_duplicate_metric_rejected(self):
        data = load_revision()
        data['metrics'].append(copy.deepcopy(data['metrics'][0]))
        self.reject(data, 'Duplicate')

    def test_staging_and_scientific_boundaries(self):
        data = load_revision()
        self.assertIn(data['publication_status'], {'STAGING_FINAL_MANIFEST_PENDING', 'PUBLICATION_AUTHORIZED'})
        self.assertEqual(next(r for r in data['requirements'] if r['id'] == 'meaningful_value_improvement')['status'], 'OPEN')
        main = [m for m in data['metrics'] if m['id'].startswith('historical_')]
        self.assertEqual(len(main), 12)
        self.assertEqual({m['cohort'] for m in main}, {'original_frozen_Test_130'})
        self.assertTrue(all(m['sd'] is not None for m in main))
        self.assertFalse(data['power']['original_implementation_verified'])
        inputs = {m['model']: m for m in data['model_inputs']}
        self.assertTrue(inputs['B']['table_text'])
        self.assertFalse(inputs['B']['image'])
        vision = next(m for m in data['metrics'] if m['id'] == 'financial_evidence_vision_baseline')
        self.assertIn('Not a zero-text model', vision['caveat'])


if __name__ == '__main__':
    unittest.main()


def test_audited_ui_all_cohorts_and_pilot_separation():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert any('25%' in message.value for message in app.info)
    for group in app.radio(key='m05_period').options:
        app.radio(key='m05_period').set_value(group).run()
        for cohort in app.selectbox(key='m05_audited_cohort').options:
            assert is_pilot(cohort) == group.startswith('Later Pilot')
            app.selectbox(key='m05_audited_cohort').set_value(cohort).run()
            assert not app.exception
    assert any('Complete means' in message.value for message in app.caption)
    assert any('Conditional power planning' in heading.value for heading in app.subheader)


def test_final_control_labels_and_all_requirement_items():
    data = load_revision()
    assert len(data['public_manifest_sha256']) == 64
    assert len(data['metrics']) == 52
    atoms = {row['id']: row for row in data['advisor_atom_crosswalk']}
    assert len(atoms) == 21
    assert atoms['R2']['status'] == 'OPEN'
    assert atoms['R18']['status'] == 'PARTIAL'
    assert atoms['R20']['status'] == 'PARTIAL'
    assert atoms['R21']['status'] == 'MET'  # Deadline recorded, not review completed.
    metrics = {row['id']: row for row in data['metrics']}
    candidate = metrics['pilot_layout_fullcoverage_text_only']
    ocr = metrics['pilot_layout_fullcoverage_ocr_pool']
    assert 'Candidate-only' in metric_table([candidate])[0]['Model / condition']
    assert 'OCR-text' in metric_table([ocr])[0]['Model / condition']
    assert round(candidate['mean'], 4) == 30.9565
    assert round(ocr['mean'], 4) == 30.2609
    assert is_pilot(candidate['cohort'])
    assert candidate['professor_compliance_evidence'] is False


def test_historical_and_decoder_remain_separate_and_scoped():
    data = load_revision()
    historical = data['historical_temporal_interpretation']
    assert historical['stored_observations']['temporal_correct'] == 126
    assert historical['stored_observations']['fusion_correct'] == 108
    assert historical['stored_observations']['targets'] == 130
    assert historical['R5_status'] == 'MATCHED_VALIDATION_CONTROL_EVIDENCED'
    assert all('id' not in source for source in historical['sources'])
    assert next(r for r in data['advisor_atom_crosswalk'] if r['id'] == 'R5')['status'] == 'MET'
    diagnostic = data['post_M5_numeric_decoder_diagnostic']
    rows = {(r['arm'], r['decoder']): r for r in diagnostic['results']}
    assert rows['no_aux', 'learned']['mean'] < rows['no_aux', 'shuffled_weights']['mean']
    assert rows['no_aux', 'learned']['accepted'] == 106
    assert rows['no_aux', 'uniform']['accepted'] == 41
    assert rows['row_column_cell_aux', 'learned']['mean'] > rows['row_column_cell_aux', 'shuffled_weights']['mean']
    assert diagnostic['scientific_outcome'] == 'GENERALIZABLE_NEURAL_VALUE_IMPROVEMENT_OPEN'
    assert 'w.roll(1)' in diagnostic['control_definition']


def test_decoder_missing_control_rejected(tmp_path):
    data = load_revision()
    data['post_M5_numeric_decoder_diagnostic']['results'].pop()
    path = tmp_path / 'revision.json'
    path.write_text(json.dumps(data))
    import pytest
    with pytest.raises(ValueError, match='paired decoder control'):
        load_revision(path)


def test_robustness_rejects_unmatched_coverage_and_missing_controls():
    import pytest

    from src.meeting05_revision import validate_robustness
    # Synthetic validation fixture, not reported scientific evidence.
    evidence = {
        'metric_columns': ['arm', 'control', 'strict_mean_pct', 'strict_seed_sd_pp', 'tie_credit_mean_pct'],
        'common_acceptance': 106, 'common_fallback': 9,
        'results': [[arm, control, 50.0, 1.0, 49.0]
                    for arm in ('no_aux', 'row_column_cell_aux')
                    for control in ('matched_uniform', 'matched_learned', 'matched_random')],
        'originalD_modified_or_evaluated': False,
    }
    for key in ('model', 'cohort', 'random_control', 'strict_ties', 'independence',
                'conclusion', 'scope_limit', 'source_hashes'):
        evidence[key] = 'Synthetic test context'
    validate_robustness(evidence)
    evidence['common_acceptance'] = 41
    with pytest.raises(ValueError, match='coverage'):
        validate_robustness(evidence)
    evidence['common_acceptance'] = 106
    evidence['results'].pop()
    with pytest.raises(ValueError, match='control'):
        validate_robustness(evidence)


def test_final_v4_robustness_is_prominent_and_preserves_history():
    from streamlit.testing.v1 import AppTest
    data = load_revision()
    assert data['canonical_sha256'] == '77574b3cedb7854708ef3a8723206f7497b9ea2ba69ea23f7dea0d6f6cbe7db0'
    assert data['public_manifest_sha256'] == 'b4d02d003a3f10cfe87d1ff13b733c2b92616ca8496e0298621ac8baebc1b438'
    assert data['publication_status'] == 'STAGING_FINAL_MANIFEST_PENDING'
    assert len(data['metrics']) == 52
    evidence = data['final_robustness_audit']
    assert evidence['originalD_modified_or_evaluated'] is False
    assert evidence['feasibility']['confirmed_unused_eligible_cohort'] is False
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert any('Final robustness correction' in h.value for h in app.subheader)
    assert any('no convincing learned grounding advantage' in w.value for w in app.warning)
    app.radio(key='m05_period').set_value('Later Pilot — exploratory appendix').run()
    assert not app.exception
    headings = [h.value for h in app.subheader]
    assert headings.index('Correction: matched-coverage numeric decoder controls') < headings.index('Earlier exploratory decoder history — unequal coverage')
    assert any('superseded for attribution' in w.value for w in app.warning)


def test_overnight_control_numerics_scope_and_render():
    import pytest
    from streamlit.testing.v1 import AppTest
    evidence = load_revision()['post_M5_CONTROL_source_hybrid_v13']
    expected = [(41.4035, 2.7320), (45.9649, 2.3700), (73.3333, 1.8190),
                (67.8947, 2.2874), (34.9123, 2.2874)]
    for row, values in zip(evidence['validation_results'], expected):
        assert row['certified_Val114_mean_sd_percent'] == pytest.approx(values, abs=0.00005)
    assert 'code_proof' not in evidence
    assert evidence['reference_status_counts']['quarantined'] == 11
    assert evidence['order_copy']['new_accuracy_measured'] is False
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    choices = app.sidebar.radio[0].options
    assert all(f'{i:02}' in choice for i, choice in zip(range(2, 6), choices))
    for choice in choices:
        app.sidebar.radio[0].set_value(choice).run()
        assert not app.exception
    assert any('CONTROL SYSTEM' in h.value for h in app.subheader)
    assert any('No neural-learning' in w.value for w in app.warning)
    tables = [frame.value for frame in app.dataframe]
    assert any(len(t) == 5 and 'Value accuracy (%)' in t for t in tables)
    assert any('all Validation 115' in e.label for e in app.expander)


def test_control_rejects_cohort_conflation_and_code():
    import pytest

    from src.meeting05_revision import validate_control
    evidence = copy.deepcopy(load_revision()['post_M5_CONTROL_source_hybrid_v13'])
    evidence['certified_counts']['Validation'] = 115
    with pytest.raises(ValueError, match='cohort'):
        validate_control(evidence)
    evidence['certified_counts']['Validation'] = 114
    evidence['code_proof'] = {}
    with pytest.raises(ValueError, match='scope'):
        validate_control(evidence)


def test_plain_progress_power_and_current_reporting_status():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert not app.get('json')
    assert not app.code
    tables = [frame.value for frame in app.dataframe]
    assert not any('literature review' in x.value.lower() for x in app.subheader)
    assert not any('Literature Review' in t.to_json() for t in tables)
    power = next(t for t in tables if 'Archived quantity' in t)
    assert dict(power.values)['Variance (squared percentage points)'] == '709.3858'
    assert any('816' in x.value and 'conditional planning' in x.value for x in app.markdown)
    assert any('650 seed-target rows are not independent' in x.value for x in app.warning)
    requirements = next(t for t in tables if 'Requirement' in t and 'Main mean±SD table' in t['Requirement'].values)
    statuses = dict(zip(requirements['Requirement'], requirements['Current status']))
    assert statuses['Main mean±SD table'] == 'Complete'
    assert statuses['Improve correct value selection'] == 'System improvement demonstrated; learned-model improvement remains open'
    assert statuses['Revisit power after above-chance progress'] == 'Conditional follow-up — not recalculated'
    assert not any('Run provenance' in e.label for e in app.expander)
