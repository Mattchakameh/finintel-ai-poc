import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.meeting05_five_seed import load_five_seed


def test_audited_source_reference_results_and_cohort_boundaries():
    data = load_five_seed()
    rows = {r['arm']: r for r in data['results'] if r['cohort'] == 'certified114_system_yield' and r['variant'] == 'metadata_assisted'}
    for arm, mean, sd in [('baseline', 28.4211, 3.0764), ('row_period', 62.8070, 2.6022), ('strict_scope_unit', 57.3684, 3.1990)]:
        assert rows[arm]['value_accuracy_percent']['mean'] == pytest.approx(mean, abs=0.00005)
        assert rows[arm]['value_accuracy_percent']['sample_sd'] == pytest.approx(sd, abs=0.00005)
    assert {r['N_per_seed'] for r in data['results']} == {113, 114, 115}
    encoded = json.dumps(data)
    for forbidden in ['/Users/', 'drive.google.com', 'source_file', '@', 'libfile_']:
        assert forbidden not in encoded


def test_consolidated_tabs_and_replay_views():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    for choice in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(choice).run()
        assert not app.exception
    assert [t.label for t in app.tabs] == ['Research Overview', 'Cohort-Specific Results', 'Methods, Requirements & Evidence']
    assert any('28.42%' in m.value and '62.81%' in m.value for m in app.markdown)
    assert any('Public links and downloads are not included' in i.value for i in app.info)
    for cohort in ['certified114_system_yield', 'all115_legacy', 'certified_candidate_available113']:
        app.selectbox(key='m05_d_replay_cohort').select(cohort).run()
        assert not app.exception


def test_private_resources_only_when_explicitly_configured(monkeypatch, tmp_path):
    path = tmp_path / 'resources.json'
    path.write_text(json.dumps({'resources': [{'name': 'Portable notebook', 'url': 'https://example.invalid/private', 'status': 'Colab runtime untested', 'sha256': 'a' * 64}], 'slides_status': 'Slides pending final hashes'}))
    monkeypatch.setenv('M05_PRIVATE_RESOURCES', str(path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert any('Colab runtime untested' in c.value for c in app.caption)


def test_neutral_labels_preserve_source_records():
    from src.meeting05_loader import load_meeting05
    from src.meeting05_revision import load_revision, metric_table
    source = load_meeting05()
    assert any('m05_dot_' in r['source'] for r in source['provenance'])
    metrics = load_revision()['metrics']
    assert any(m['label_version'] == 'dot_challenge_source_verified' for m in metrics)
    assert 'dot_challenge' not in json.dumps(metric_table(metrics))
    assert 'M05_dot_' not in json.dumps(metric_table(metrics))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    for frame in app.dataframe:
        assert 'm05_dot_' not in frame.value.to_json()
    assert not any('SHA-256' in e.label for e in app.expander)


def test_public_snapshot_excludes_private_resource_urls():
    from src.meeting05_loader import load_meeting05
    summary = load_meeting05()
    assert len(summary['links']) == 6
    assert set(summary['links'].values()) == {'PRIVATE_RESOURCE_NOT_INCLUDED'}
    encoded = json.dumps(summary)
    assert 'google.com/' not in encoded
    assert '/Users/' not in encoded


def test_integrated_overview_keeps_current_history_and_pilot_distinct():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    headings = [h.value for h in app.subheader]
    assert headings.index('Meeting 05 results — source-assisted comparison') < headings.index('Historical A–D comparison (Meeting4)')
    assert 'What is done — and what remains' in headings
    assert any('not a confidence interval' in c.value for c in app.caption)
    assert any('A row near 23.85% is model A' in c.value for c in app.caption)
    assert any(e.label == 'Explore separate cohorts and earlier experiments' for e in app.expander)
    app.radio(key='m05_period').set_value('Later Pilot — exploratory appendix').run()
    assert not app.exception
    assert any('Later pilot — exploratory system diagnostics' == h.value for h in app.subheader)
    assert not app.get('json')


def test_compatibility_receipt_is_not_accuracy_or_new_training():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    text = ' '.join(x.value for x in list(app.markdown) + list(app.caption))
    assert '575/575 archived value choices, not 100% accuracy' in text
    assert '62.8070% ± 2.6022' in text
    assert 'No bitwise-logit equivalence' in text
    assert 'Colab runtime has not been tested' not in text
    from src.meeting05_priorities import PRIORITIES
    assert len(PRIORITIES) == 5
    assert all(p['question'].endswith('?') and p['answer'] for p in PRIORITIES)
