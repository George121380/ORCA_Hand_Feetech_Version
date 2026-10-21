"""Local finite-response fitting; not a model of global grasp reachability."""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def pose_delta(position,rotation,reference_position,reference_rotation):
    """World translation in meters and world rotation vector in radians."""
    return np.r_[np.asarray(position)-reference_position,
                 Rotation.from_matrix(np.asarray(rotation)@np.asarray(reference_rotation).T).as_rotvec()]


def local_pose_metrics(position,rotation,start_position,start_rotation,target_deg=2.):
    increment=Rotation.from_rotvec([0,0,np.deg2rad(target_deg)]).as_matrix()
    motion=pose_delta(position,rotation,start_position,start_rotation)
    error=Rotation.from_matrix(increment@np.asarray(start_rotation)@np.asarray(rotation).T).magnitude()
    displacement=float(np.linalg.norm(motion[:3]))
    return dict(world_rotvec_deg=np.rad2deg(motion[3:]).tolist(),
        off_axis_rotation_deg=float(np.linalg.norm(np.rad2deg(motion[3:5]))),
        position_displacement_m=displacement,target_error_deg=float(np.rad2deg(error)),
        local_pose_tracking_met=bool(error<np.deg2rad(.5) and displacement<.001))


def fit_bounded_response(jacobian,desired_delta,bound=.04,regularization=.1):
    """Fit only uncensored columns; keep invalid motor channels at zero.

    Translation is weighted per1mm and rotation per1deg. The small positive
    ridge favors smaller residual actions; it never changes the action bound.
    """
    matrix=np.asarray(jacobian,dtype=float)
    desired=np.asarray(desired_delta,dtype=float)
    if matrix.shape!=(6,16) or desired.shape!=(6,) or not np.isfinite(desired).all():
        raise ValueError('Expected6x16 local response and a finite6D target')
    if not 0<bound<=.04 or not 0<regularization<=1:
        raise ValueError('Local diagnostic bound/regularization invalid')
    valid=np.flatnonzero(np.isfinite(matrix).all(axis=0))
    invalid=np.flatnonzero(~np.isfinite(matrix).all(axis=0))
    scales=np.r_[np.full(3,1000.),np.full(3,180/np.pi)]
    residual=np.zeros(16)
    predicted=np.zeros(6)
    summary=dict(valid_columns=valid.tolist(),censored_columns=invalid.tolist(),
                 scales=scales.tolist(),bound=bound,regularization=regularization)
    if not len(valid):
        return dict(summary,solver_success=False,rank=0,singular_values=[],
                    residual_action=residual.tolist(),predicted_delta=predicted.tolist(),
                    reason='No uncensored full-duration response pairs')
    weighted=scales[:,None]*matrix[:,valid]
    values=np.linalg.svd(weighted,compute_uv=False)
    rank=int(np.sum(values>max(float(values[0]),1e-12)*1e-6))
    augmented=np.vstack([weighted,regularization*np.eye(len(valid))])
    target=np.r_[scales*desired,np.zeros(len(valid))]
    solution=lsq_linear(augmented,target,bounds=(-bound,bound),tol=1e-10,max_iter=200)
    residual[valid]=solution.x
    predicted=matrix[:,valid]@solution.x
    return dict(summary,solver_success=bool(solution.success),rank=rank,singular_values=values.tolist(),
                residual_action=residual.tolist(),predicted_delta=predicted.tolist(),
                weighted_tracking_error=float(np.linalg.norm(scales*(predicted-desired))),
                residual_norm=float(np.linalg.norm(residual)),solver_message=solution.message)
