import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('local_validation',ROOT/'scripts/validate_rotation_local_candidate.py')
validation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_rejects_modified_parent_evidence_before_using_its_fit(tmp_path):
    plan=json.loads((ROOT/'configs/training/pen-rotate-local-validation-v2.json').read_text())
    parent=tmp_path/'changed.json'
    parent.write_text('{"status":"complete","fit":{"rank":6}}')
    with pytest.raises(ValueError,match='exact immutable33-case'):
        validation.load_parent(parent,plan)


@pytest.mark.parametrize('key,value',[
    ('maximum_cases',2),('local_target_deg',.5),('residual_action_bound',.08),
    ('seed',2611000),('parent_report_sha256','other'),('pulse_ticks',20),
])
def test_rejects_silent_budget_target_bound_seed_or_parent_changes(tmp_path,key,value):
    plan=json.loads((ROOT/'configs/training/pen-rotate-local-validation-v2.json').read_text())
    plan[key]=value
    with pytest.raises(ValueError,match='Changed local protocol'):
        validation.load_parent(tmp_path/'not-read.json',plan)
