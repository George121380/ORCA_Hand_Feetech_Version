"""A failed long job must retain its status and never proceed to evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("training_job", Path(__file__).parents[1] / "scripts/run_training_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def test_grasp_stage_must_meet_original_threshold_and_match_hold_baseline():
    policy=dict(success_rate=.8,drop_rate=0.,acceptance_met=True)
    assert not job.grasp_acceptance(policy,dict(success_rate=.9,drop_rate=0.))[0]
    assert job.grasp_acceptance(policy,dict(success_rate=.7,drop_rate=0.))[0]
    assert not job.grasp_acceptance(dict(policy,drop_rate=.01),dict(success_rate=.7,drop_rate=0.))[0]
    assert not job.grasp_acceptance(dict(policy,acceptance_met=False),dict(success_rate=.7,drop_rate=0.))[0]


def test_selection_and_test_seeds_cannot_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output),
                       "--selection-seed", "20050", "--test-seed", "20000"])
    with pytest.raises(SystemExit):
        job.main()
    assert not output.exists()


def test_new_grasp_job_forwards_explicit_cli_and_persists_family(monkeypatch,tmp_path):
    monkeypatch.setattr(job.platform,'system',lambda:'Linux')
    output=tmp_path/'grasp'
    calls=[]
    class Failed:
        pid=99999998
        def __init__(self,argv,**kwargs):calls.append(argv)
        def wait(self):return 7
    monkeypatch.setattr(job.subprocess,'Popen',Failed)
    monkeypatch.setattr(job.sys,'argv',['job','--output',str(output),'--task-family','pen-grasp-v1',
        '--object','pen','--angle','10','--horizon','140','--randomization','0','--zero-initial-action-mean'])
    with pytest.raises(RuntimeError,match='train exited 7'):
        job.main()
    assert Path(calls[0][1]).name=='pen_grasp.py'
    assert '--zero-initial-action-mean' in calls[0]
    state=json.loads((output/'job.json').read_text())
    assert state['arguments']['task_family']=='pen-grasp-v1' and not state['acceptance_met']
    assert state['arguments']['zero_initial_action_mean']


def test_rotation_job_initializes_once_then_recovers_rotation_checkpoint(monkeypatch,tmp_path):
    monkeypatch.setattr(job.platform,'system',lambda:'Linux')
    monkeypatch.setattr('orca_feetech.pen_rotate.validate_grasp_prefix',lambda path:None)
    output=tmp_path/'rotation'
    calls=[]
    class Failed:
        pid=99999998
        def __init__(self,argv,**kwargs):calls.append(argv)
        def wait(self):return 7
    monkeypatch.setattr(job.subprocess,'Popen',Failed)
    monkeypatch.setattr(job.sys,'argv',['job','--output',str(output),'--task-family','pen-rotate-v1',
        '--angle','15','--horizon','140','--initialize-policy','/frozen/grasp/policy.zip'])
    with pytest.raises(RuntimeError,match='train exited 7'):
        job.main()
    assert Path(calls[0][1]).name=='pen_rotate.py'
    assert calls[0][calls[0].index('--initialize-policy')+1]=='/frozen/grasp/policy.zip'
    (output/'model').mkdir()
    checkpoint=output/'model/checkpoint_20000_steps.zip'
    monkeypatch.setattr(job,'running_processes',lambda *a:[])
    monkeypatch.setattr(job,'latest_checkpoint',lambda *a,**kw:(checkpoint,20000))
    monkeypatch.setattr(job.sys,'argv',['job','--output',str(output),'--continue-existing'])
    with pytest.raises(RuntimeError,match='train-recovery-1 exited 7'):
        job.main()
    assert '--initialize-policy' not in calls[1]
    assert calls[1][calls[1].index('--resume')+1]==str(checkpoint)


def test_training_failure_stops_job_and_preserves_command(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output)])
    calls = []

    class FailedProcess:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 7

    monkeypatch.setattr(job.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="train exited 7"):
        job.main()
    state = json.loads((output / "job.json").read_text())
    assert len(calls) == 1
    assert state["status"] == "failed"
    assert state["commands"][0]["returncode"] == 7
    assert not state["acceptance_met"]


def test_prefix_teacher_is_forwarded_and_cannot_change_on_recovery(monkeypatch,tmp_path):
    monkeypatch.setattr(job.platform,"system",lambda:"Linux")
    teacher = tmp_path/"teacher.zip"
    teacher.write_bytes(b"unloaded command fixture")
    (tmp_path/"config.json").write_text("{}")
    output = tmp_path/"job"
    calls = []
    class FailedProcess:
        pid = 99999998
        def __init__(self,command,**kwargs):
            calls.append(command)
        def wait(self):
            return 7
    monkeypatch.setattr(job.subprocess,"Popen",FailedProcess)
    monkeypatch.setattr(job.sys,"argv",["job","--output",str(output),
        "--angle","30","--turns","2","--horizon","480","--randomization","1",
        "--prefix-reset-checkpoint",str(teacher),"--prefix-reset-probability","0.75"])
    with pytest.raises(RuntimeError,match="train exited 7"):
        job.main()
    assert calls[0][calls[0].index("--prefix-reset-checkpoint")+1] == str(teacher)
    assert calls[0][calls[0].index("--prefix-reset-probability")+1] == "0.75"
    state = json.loads((output/"job.json").read_text())
    assert state["prefix_reset_source_sha256"] == job.sha256(teacher)
    teacher.write_bytes(b"changed teacher")
    monkeypatch.setattr(job,"running_processes",lambda *a:[])
    monkeypatch.setattr(job.sys,"argv",["job","--output",str(output),"--continue-existing"])
    with pytest.raises(SystemExit):
        job.main()
    assert len(calls) == 1


def test_recovery_keeps_remaining_budget_and_reduces_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    (output / "model").mkdir(parents=True)
    arguments = dict(output=str(output), object="pen", angle=15, turns=1, horizon=240,
                     randomization=.3, steps=3000000, num_envs=16, device="cuda", seed=0,
                     resume=None, device_config=None, mass_kg=None, friction=1.,
                     selection_episodes=40, selection_seed=30000, test_episodes=100,
                     test_seed=20000, video=False, reset_log_std=-1., learning_rate=.0001,
                     entropy_coef=.001, target_kl=.01)
    (output / "job.json").write_text(json.dumps(dict(status="failed", arguments=arguments,
                                                    pid=99999999, commands=[], acceptance_met=False)))
    source = output / "model/checkpoint_1500000_steps.zip"
    monkeypatch.setattr(job, "latest_checkpoint", lambda *args: (source, 1500000))
    calls = []

    class FailedProcess:
        pid = 99999998

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 7

    monkeypatch.setattr(job.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output), "--continue-existing",
                       "--recovery-num-envs", "8"])
    with pytest.raises(RuntimeError, match="train-recovery-1 exited 7"):
        job.main()
    command = calls[0]
    assert command[command.index("--steps") + 1] == "1500000"
    assert command[command.index("--resume") + 1] == str(source)
    assert command[command.index("--num-envs") + 1] == "8"
    assert command[command.index("--learning-rate") + 1] == "0.0001"
    assert "--reset-log-std" not in command
    state = json.loads((output / "job.json").read_text())
    assert state["target_total_steps"] == 3000000
    assert state["arguments"]["steps"] == 3000000
    assert (output / "current-training.log").readlink() == Path("train-recovery-1.log")


@pytest.mark.parametrize("initial_passes", [False, True])
def test_curriculum_stops_at_first_failed_gate(monkeypatch, tmp_path, initial_passes):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location("pen_course", Path(__file__).parents[1] / "scripts/run_pen_curriculum.py")
    course = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(course)
    monkeypatch.setattr(course.platform, "system", lambda: "Linux")
    initial = tmp_path / "initial"
    initial.mkdir()
    arguments = dict(object="pen", angle=15, turns=1, randomization=.3, selection_seed=30000,
                     test_seed=20000, test_episodes=100, num_envs=16, device="cuda", seed=0)
    (initial / "job.json").write_text(json.dumps(dict(status="complete", acceptance_met=initial_passes,
                                                    arguments=arguments)))
    output = tmp_path / "course"
    monkeypatch.setattr(course.sys, "argv", ["course", "--initial-job", str(initial), "--output", str(output)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        child = Path(command[command.index("--output") + 1])
        child.mkdir()
        (child / "job.json").write_text(json.dumps(dict(status="complete", acceptance_met=False, test={})))

    monkeypatch.setattr(course.subprocess, "run", run)
    course.main()
    state = json.loads((output / "curriculum.json").read_text())
    assert state["status"] == "quality_gate_failed"
    assert len(calls) == int(initial_passes)
    if calls:
        command = calls[0]
        assert command[command.index("--test-seed") + 1] == "21000"
        assert command[command.index("--resume") + 1] == str(initial / "selected/policy.zip")
