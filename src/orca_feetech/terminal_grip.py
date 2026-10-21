"""Small terminal command perturbations; static contacts grant no task credit."""
import numpy as np


def terminal_knots(original,delta,initial,lower,upper):
    original,delta,initial,lower,upper=map(np.asarray,(original,delta,initial,lower,upper))
    if (original.shape!=(4,16) or delta.shape!=(16,) or initial.shape!=(16,)
            or not np.isfinite([*original,delta,initial]).all() or max(abs(delta))>.08+1e-12):
        raise ValueError('Expected four16D knots and finite residual<=0.08rad')
    result=original+np.array([0.,0.,.5,1.])[:,None]*delta
    if (np.any(result<lower-1e-10) or np.any(result>upper+1e-10)
            or np.max(abs(result-initial))>.3+1e-10):
        raise ValueError('Terminal command exceeds original ROM or0.3rad total delta')
    return result


def static_grip_score(normal_dot,geometry,delta):
    if (normal_dot is None or not np.isfinite(normal_dot) or normal_dot>-.5
            or not np.isfinite([geometry['self_penetration_m'],geometry['object_penetration_m']]).all()
            or geometry['excluded_support'] or max(geometry['self_penetration_m'],geometry['object_penetration_m'])>.002):
        return None
    return float((max(0.,normal_dot+.7)/.2)**2+.05*np.mean((np.asarray(delta)/.08)**2)
                 +.1*(geometry['object_penetration_m']/.002)**2)
