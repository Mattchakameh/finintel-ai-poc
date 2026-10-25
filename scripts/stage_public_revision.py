"""Import only a producer-approved public manifest; never publish automatically."""
import hashlib
import json
import sys
from pathlib import Path


def stage(source, destination):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    for word in ('/Users/', 'drive.google.com', 'libfile_', 'download_url', 'source_file'):
        if word in raw.decode():
            raise ValueError('Public manifest contains private metadata')
    historical = data.get('historical_temporal_interpretation', {})
    if historical:
        historical['sources'] = [{k: row[k] for k in ('name', 'sha256', 'bytes')} for row in historical['sources']]
    data['schema_version'] = 1
    data['publication_status'] = 'STAGING_FINAL_MANIFEST_PENDING'
    data['public_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    status_map = {
        'REPORTING_CORRECTED': 'MET',
        'MATCHED_VALIDATION_CONTROL_EVIDENCED': 'MET',
        'LOCAL_ANNOTATION_READY_EXTERNAL_WRITES_PAUSED': 'PARTIAL',
        'SCIENTIFIC_OUTCOME_OPEN': 'OPEN',
        'DOCUMENTED_WITH_LIMITATIONS': 'MET',
        'SCOPED_IMPLEMENTATION_ONLY': 'MET',
        'SCOPED_COMPARISON_COMPLETE_TEST_EXPLANATION_OPEN': 'PARTIAL',
        'EXPERIMENT_COMPLETE_INPUT_SCOPE_DISCLOSED': 'MET',
        'EXPERIMENT_COMPLETED_NEGATIVE': 'MET',
        'REPORTING_CORRECTED_H1_NOT_SUPPORTED': 'MET',
        'DOCUMENTED': 'MET',
        'EXPERIMENTS_COMPLETE_SCOPED': 'MET',
        'REPORTING_CORRECTED_CONDITIONAL': 'MET',
        'PROGRESS_REPORTING_MET_COMPLETION_OPEN': 'PARTIAL',
        'LOCAL_READY_EXTERNAL_WRITEBACK_PENDING': 'PARTIAL',
        'CLARIFICATION_MET_DEPLOYMENT_LIMITATIONS_RETAINED': 'MET',
        'CONDITIONAL_RECALCULATION_OPEN_REPORTING_CLARIFIED': 'PARTIAL',
        'DEADLINE_RECORDED_2026_11_30_NOT_OVERDUE': 'MET',
    }
    data['requirements'] = [
        {'id': r['id'], 'status': status_map[r['status']],
         'review_state': r['status'], 'scope': r['remediation']}
        for r in data.pop('requirement_statuses')]
    for row in data['advisor_atom_crosswalk']:
        row['status'] = status_map[row['current_status']]
    data['requirements_note'] = ('MET denotes a completed scoped experiment or reporting correction, '
        'including negative results; it does not mean scientific success. PARTIAL distinguishes '
        'completed progress reporting from open completion or explanation. Publication remains on hold.')
    Path(destination).write_text(json.dumps(data, indent=2)+'\n')
    print(data['public_manifest_sha256'])


if __name__ == '__main__':
    stage(*sys.argv[1:])
