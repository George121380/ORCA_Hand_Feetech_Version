from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orca_feetech.recorded_rotation_audit import audit_recorded_rotation,contact_gate


def record():
    contacts=dict(excluded=[],hand_contacts=[
        dict(body='right_thumb_dp',normal_world=[1,0,0],force_world_n=[.1,0,0]),
        dict(body='right_middle_ip',normal_world=[-1,0,0],force_world_n=[-.1,0,0])])
    frame=np.zeros(74)
    frame[47:56]=np.eye(3).ravel()
    samples=[]
    for index in range(140):
        rotation=Rotation.from_euler('z',15 if index<120 else 0,degrees=True)
        samples.append(dict(tick=index+1,observation=np.tile(frame,4).tolist(),action=[0]*16,
            qpos_all=[0]*20+rotation.as_quat(scalar_first=True).tolist(),qvel_all=[0]*23,
            command=[0]*17,command_limits_met=True,rotation=rotation.as_matrix().tolist(),position=[0]*3,
            angle_error_deg=15 if index<120 else 0,linear_speed_mps=0,angular_speed_rps=0,
            contacts=deepcopy(contacts),object_weight_n=1.,loaded_opposition=True,
            peak_self_penetration_m=0,peak_object_penetration_m=0,peak_normal_force_n=.2,
            maximum_displacement_m=0,dropped=False,excluded_force_support=False,vision_age=0))
    return dict(samples=samples,goal_rotation=np.eye(3).tolist(),start_command=[0]*17,
        inherited_command_velocity=[0]*16,initial=dict(prefix_grasp_success=True),
        end=dict(termination_reason='success',dropped=False,is_success=True),
        full_uncensored_duration=True,command_limits_met=True,simulated_seconds=7.,native_task_success=True)


def test_native_audit_checks_last_full_second_and_preserves_2ms_peaks():
    row=record()
    assert audit_recorded_rotation(row,np.full(17,-1),np.ones(17))['native_task_success']
    # One failed loaded tick invalidates the entire final window, even if the
    # final tick/pose look successful. Old native flags must be rejected.
    row['samples'][-2]['contacts']['hand_contacts'][1]['force_world_n']=[0,0,0]
    row['samples'][-2]['loaded_opposition']=False
    with pytest.raises(ValueError,match='Native success flag'):
        audit_recorded_rotation(row,np.full(17,-1),np.ones(17))
    row=record()
    row['samples'][12]['peak_object_penetration_m']=.0021
    with pytest.raises(ValueError,match='Native success flag'):
        audit_recorded_rotation(row,np.full(17,-1),np.ones(17))


def test_zero_force_opposed_normals_and_unobservable_goal_changes_are_rejected():
    row=record()
    contact=row['samples'][0]['contacts']
    contact['hand_contacts'][0]['force_world_n']=[0,0,0]
    assert not contact_gate(contact,1.)
    row=record()
    row['samples'][0]['observation'][-27]=.5
    with pytest.raises(AssertionError):
        audit_recorded_rotation(row,np.full(17,-1),np.ones(17))
