import json
from pathlib import Path

import pytest

from src.meeting05_loader import Meeting05EvidenceError, load_meeting05


def test_snapshot_boundaries():
    data = load_meeting05()
    assert len(data['items']) == 21
    assert 21 not in [x['number'] for x in data['items']]
    results = {r['id']: r for r in data['results']}
    assert results['corrected115']['rows'][0]['Exact-choice %'] == pytest.approx(41.04347826086957)
    assert results['corrected115']['rows'][1]['Exact-choice %'] == pytest.approx(53.56521739130435)
    assert results['concept80']['rows'][1]['Exact-choice %'] < results['concept80']['rows'][0]['Exact-choice %']
    assert results['source105']['rows'][1]['Exact-choice %'] == pytest.approx(104 / 105 * 100)
    assert 'NOT 99%' in results['source105']['limit']
    assert '0/80 coverage' in results['parser80']['limit']
    assert len(data['models']) == 5  # four original configurations plus distinct M05 head


def test_missing_summary(tmp_path):
    with pytest.raises(Meeting05EvidenceError, match='unavailable'):
        load_meeting05(tmp_path / 'missing.json')


@pytest.mark.parametrize('content', ['not JSON', '{}', '{"schema_version": 999}', 'null'])
def test_corrupt_summary(tmp_path, content):
    path = tmp_path / 'summary.json'
    path.write_text(content)
    with pytest.raises(Meeting05EvidenceError):
        load_meeting05(path)


def test_out_of_range_rejected(tmp_path):
    data = load_meeting05()
    data['results'][0]['rows'][0]['Exact-choice %'] = 101
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(data))
    with pytest.raises(Meeting05EvidenceError, match='outside'):
        load_meeting05(path)


def test_repeated_navigation_and_all_cohorts():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    titles = {
        0: 'FinIntel AI -- Multimodal Financial Intelligence POC',
        1: 'FinIntel AI — Meeting 03',
        2: 'FinIntel AI — Meeting 04',
        3: 'FinIntel AI — Meeting 05',
    }
    assert not app.exception
    assert app.title[0].value == titles[0]
    choices = app.sidebar.radio[0].options
    for index in [1, 2, 3, 0, 3, 1, 0, 2, 3]:
        app.sidebar.radio[0].set_value(choices[index]).run()
        assert not app.exception, str(app.exception)
        assert app.title[0].value == titles[index]
        assert len(app.title) == 1
    for cohort in ['corrected115', 'original115', 'period20', 'concept80', 'parser80', 'source105']:
        app.selectbox(key='m05_cohort').set_value(cohort).run()
        assert not app.exception
        assert any('Cohort:' in x.value for x in app.markdown)


def test_missing_m05_evidence_keeps_navigation(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    import src.meeting05_loader as loader
    monkeypatch.setattr(loader, 'SUMMARY_PATH', tmp_path / 'missing.json')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    choices = app.sidebar.radio[0].options
    app.sidebar.radio[0].set_value(choices[3]).run()
    assert not app.exception
    assert any('evidence unavailable' in x.value for x in app.error)
    app.sidebar.radio[0].set_value(choices[1]).run()
    assert not app.exception
    assert app.title[0].value == 'FinIntel AI — Meeting 03'


def test_missing_m04_visuals_are_nonfatal(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    import src.meeting04_view as view
    monkeypatch.setattr(view, 'M04_VISUAL_DIR', tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[2]).run()
    assert not app.exception
    assert any('No Meeting 04 SEC visual assets' in x.value for x in app.warning)
