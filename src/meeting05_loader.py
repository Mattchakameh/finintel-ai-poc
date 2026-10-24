"""Load the compact, reviewed Meeting 05 evidence snapshot; never synthesize results."""
import json
from pathlib import Path

SUMMARY_PATH = Path(__file__).resolve().parents[1] / 'data/meeting05/summary.json'


class Meeting05EvidenceError(ValueError):
    pass


def load_meeting05(path=None):
    try:
        data = json.loads(Path(path or SUMMARY_PATH).read_text(encoding='utf-8'))
        if data['schema_version'] != 1:
            raise ValueError('Unsupported summary schema')
        for key in ('models', 'results', 'items', 'links', 'provenance'):
            if not data[key]:
                raise ValueError(f'Missing evidence section: {key}')
        ids = [r['id'] for r in data['results']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate result cohort IDs')
        for result in data['results']:
            for key in ('title', 'cohort', 'regime', 'limit'):
                if not result[key]:
                    raise ValueError(f'Missing result context: {key}')
            for row in result['rows']:
                if not 0 <= row['Exact-choice %'] <= 100:
                    raise ValueError('Accuracy outside 0–100')
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Meeting05EvidenceError(f'Meeting 05 evidence unavailable or invalid: {exc}') from exc
