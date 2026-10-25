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
    assert any('MET denotes' in message.value for message in app.caption)
    assert any('progress' in heading.value.lower() for heading in app.subheader)


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
