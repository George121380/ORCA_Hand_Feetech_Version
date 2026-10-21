"""Finite position trajectory and cost for a privileged development search."""
import numpy as np


OFFSET_LIMIT=.18


def trajectory_target(initial,offsets,seconds,lower,upper):
    offsets=np.asarray(offsets,dtype=float)
    if offsets.shape!=(2,16) or not np.isfinite(offsets).all() or np.max(np.abs(offsets))>OFFSET_LIMIT+1e-12:
        raise ValueError('Require two finite16D offsets within0.18rad')
    if not np.isfinite(seconds) or not 0<=seconds<=7.+1e-10:
        raise ValueError('Trajectory time must be within the original7s horizon')
    initial=np.asarray(initial)[:16]
    knots=np.clip(initial+offsets,np.asarray(lower)[:16],np.asarray(upper)[:16])
    if seconds<=2.5:
        start,end=initial,knots[0]
        phase=seconds/2.5
    else:
        start,end=knots
        phase=np.clip((seconds-2.5)/2.5,0,1)
    blend=phase**3*(10+phase*(-15+6*phase))
    return start+blend*(end-start)


def trajectory_cost(samples,offsets,*,complete,command_limits_met):
    """Physical failures cannot beat a complete, valid low-error candidate.

    Native is_success is supplied by the unchanged environment. No substitute
    local target, transient goal hit or grasp-prefix success is credited here.
    """
    final=samples[-20:]
    if not final:
        return 10000.
    score=np.mean([(s['angle_error_deg']/15.)**2
        +.05*min((s['linear_speed_mps']/.015)**2,100.)
        +.05*min((s['angular_speed_rps']/.3)**2,100.)
        +2.*(not s['loaded_opposition'])+.1*(s['handoff_displacement_m']/.01)**2 for s in final])
    score+=.01*np.mean((np.asarray(offsets)/OFFSET_LIMIT)**2)
    if not complete or not command_limits_met:
        score+=10000.
    elif final[-1]['is_success']:
        score-=100.
    return float(score)
