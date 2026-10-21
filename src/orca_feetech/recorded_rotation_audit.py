"""Regrade recorded end windows and cumulative audits without simulating.

The2ms contact peaks were captured by the environment, not reconstructed from
20Hz snapshots. This audit checks those records and observable training pairs.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def contact_gate(snapshot,weight):
    forces={}
    normals={}
    for c in snapshot['hand_contacts']:
        force=max(0.,float(np.asarray(c['force_world_n'])@c['normal_world']))
        forces[c['body']]=forces.get(c['body'],0.)+force
        if force>1e-6:
            normals.setdefault(c['body'],[]).append(np.asarray(c['normal_world']))
    thumb,middle='right_thumb_dp','right_middle_ip'
    dots=[float(a@b) for a in normals.get(thumb,[]) for b in normals.get(middle,[])]
    return bool(min(forces.get(thumb,0.),forces.get(middle,0.))>=.05*weight
                and dots and min(dots)<=-.5)


def audit_recorded_rotation(row,lower,upper):
    samples=row['samples']
    goal=np.asarray(row['goal_rotation'])
    previous=np.asarray(row['start_command'])
    velocity=np.asarray(row['inherited_command_velocity'])
    peak_v=peak_a=0.
    hard_ok=command_ok=True
    final_gates=[]
    for index,s in enumerate(samples):
        if s['tick']!=index+1:
            raise ValueError('Nonconsecutive recorded control ticks')
        obs,action=np.asarray(s['observation']),np.asarray(s['action'])
        qpos,qvel=np.asarray(s['qpos_all']),np.asarray(s['qvel_all'])
        command=np.asarray(s['command'])
        if (obs.shape!=(296,) or action.shape!=(16,) or qpos.shape!=(24,) or qvel.shape!=(23,)
                or not all(np.isfinite(a).all() for a in (obs,action,qpos,qvel,command))
                or np.max(abs(action))>1+1e-10):
            raise ValueError('Malformed observable/state/action record')
        np.testing.assert_allclose(obs.reshape(4,74)[-1,47:56],goal.ravel(),rtol=0,atol=6e-8)
        np.testing.assert_allclose(qpos[17:20],s['position'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(Rotation.from_quat(qpos[20:24],scalar_first=True).as_matrix(),
                                   s['rotation'],rtol=0,atol=1e-10)
        new_velocity=(command[:16]-previous[:16])/.05
        peak_v=max(peak_v,float(max(abs(new_velocity))))
        peak_a=max(peak_a,float(max(abs(new_velocity-velocity)))/.05)
        command_ok &= bool(np.all(command>=lower-1e-10) and np.all(command<=upper+1e-10)
            and abs(command[-1]-row['start_command'][-1])<1e-10 and peak_v<=.8+1e-9 and peak_a<=2+1e-9)
        if command_ok!=s['command_limits_met']:
            raise ValueError('Recorded command gate differs from reconstructed controller limits')
        previous,velocity=command,new_velocity
        error=float(np.rad2deg(Rotation.from_matrix(np.asarray(s['rotation'])@goal.T).magnitude()))
        np.testing.assert_allclose(error,s['angle_error_deg'],rtol=0,atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(qvel[17:20]),s['linear_speed_mps'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(qvel[20:23]),s['angular_speed_rps'],rtol=0,atol=1e-10)
        load=contact_gate(s['contacts'],s['object_weight_n'])
        if load!=s['loaded_opposition']:
            raise ValueError('Stored loaded opposition differs from positive-force contact normals')
        hard_ok &= bool(max(s['peak_self_penetration_m'],s['peak_object_penetration_m'])<=.002
            and s['peak_normal_force_n']<=10*s['object_weight_n'] and s['maximum_displacement_m']<.04
            and not s['dropped'] and not s['excluded_force_support'] and s['vision_age']<=.25)
        final_gates.append(bool(hard_ok and error<8 and load and s['contacts']['hand_contacts']
            and s['linear_speed_mps']<.015 and s['angular_speed_rps']<.3))
    complete=bool(row['initial']['prefix_grasp_success'] and len(samples)==140
        and row['end']['termination_reason'] in ('horizon','success') and not row['end']['dropped'])
    if complete!=row['full_uncensored_duration'] or command_ok!=row['command_limits_met']:
        raise ValueError('Stored duration/command labels differ from raw records')
    if complete and not np.isclose(row['simulated_seconds'],7.,atol=1e-9,rtol=0):
        raise ValueError('Full continuation is not7s')
    angular=distance=None
    if len(samples)>=20:
        last=samples[-1]
        angular=max(float(np.rad2deg(Rotation.from_matrix(
            np.asarray(s['rotation'])@np.asarray(last['rotation']).T).magnitude())) for s in samples[-20:])
        distance=max(float(np.linalg.norm(np.asarray(s['position'])-last['position'])) for s in samples[-20:])
    success=bool(complete and command_ok and hard_ok and all(final_gates[-20:])
        and len(final_gates)>=20 and angular<2 and distance<.002)
    if success!=row['native_task_success'] or success!=row['end']['is_success']:
        raise ValueError('Native success flag disagrees with recorded physical gates')
    return dict(native_task_success=success,observable_action_pairs=len(samples),
        final_window_valid_ticks=sum(final_gates[-20:]),final_orientation_span_deg=angular,
        final_position_span_m=distance,peak_command_velocity_rad_s=peak_v,
        peak_command_acceleration_rad_s2=peak_a,recorded_cumulative_2ms_limits_met=hard_ok)
