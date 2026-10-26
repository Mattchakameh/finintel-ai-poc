"""Presentation mapping supplied by the audited slide coordination; anchors pending."""
# Internal notebook items are subordinate to these five advisor priorities.
PRIORITIES = [
    {'number': 1, 'title': 'Chance-level results and correct-value progress',
     'items': '1, 6, 12', 'slides': '2, 3, 7, 8, 19, 23, 28, 49–51',
     'rows': [('R1', '1', '2, 3, 7, 8, 19, 23, 28, 49–51'),
              ('R2', '6', '2, 3, 7, 8, 19, 23, 28, 49–51')],
     'note': 'Internal item 12 (tiling) supports priorities 1 and 3; it is not a sixth professor priority. System gains do not establish learned-model improvement.'},
    {'number': 2, 'title': 'Candidate construction and reporting-year evidence',
     'items': '2, 3, 4, 5', 'slides': '4, 5, 6, 40',
     'rows': [('R3', '2', '5'), ('R4', '5', '6, 40'), ('R5', '3–4', '4')],
     'note': 'Separate actual year availability from the matching-year rule and learned-selector comparison.'},
    {'number': 3, 'title': 'Model inputs, modality controls and H1 interpretation',
     'items': '7, 8, 9, 10, 11; supporting item 12', 'slides': '3, 9, 10, 11, 27',
     'rows': [('R10', '7', '9'), ('R6', '8', '10'), ('R7', '9', '11'),
              ('R8', '10', '27'), ('R9', '11', '3, 27')],
     'note': 'Internal item 12 (tiling) also supports this priority. Gate weights are not causal importance; negative findings and H1 limits remain.'},
    {'number': 4, 'title': 'Harder candidates and performance by error type',
     'items': '13, 14, 15, 16, 17', 'slides': '15–18, 29–31, 35, 42–44',
     'rows': [('R11', '13', '29'), ('R12', '14', '30'),
              ('R13', '15', '15: historical 29 targets; 16, 42: new 20 Validation targets'),
              ('R14', '16', '31'), ('R15', '17', '17–18, 35, 43–44')],
     'note': 'The historical 29-target period challenge and new 20-target Validation challenge are different cohorts; do not pool their results.'},
    {'number': 5, 'title': 'Mean/SD reporting, exclusions and conditional power',
     'items': '18, 19, 20', 'slides': '2, 3, 23, 32–34',
     'rows': [('R16', '18', '2, 3, 23, 32'), ('R19', '19', '33'),
              ('R17', '20', '34'), ('R18', '20', '34')],
     'note': 'Mean/SD reporting is complete. The approximately 816-target figure is archived conditional planning; a new power calculation has not been performed.'},
]


def private_priority_links():
    """Resolve anchors only from the optional private configuration, never public data."""
    import json
    import os
    from pathlib import Path
    config = os.environ.get('M05_PRIVATE_RESOURCES')
    if not config:
        return {}
    resources = json.loads(Path(config).read_text())['resources']
    notebook = next((r for r in resources if r.get('name', '').startswith('Notebook')), {})
    return notebook.get('priority_links', {})
