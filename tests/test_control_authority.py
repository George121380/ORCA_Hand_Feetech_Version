import numpy as np
from scipy.spatial.transform import Rotation

from orca_feetech.control_authority import pose_delta,local_pose_metrics,fit_bounded_response


def test_local_motion_and_target_are_world_frame_with_rotated_reference():
    reference=Rotation.from_euler('x',90,degrees=True).as_matrix()
    target=Rotation.from_euler('z',2,degrees=True).as_matrix()@reference
    response=pose_delta(np.array([.1,.2,.3]),target,np.array([.1,.2,.3]),reference)
    np.testing.assert_allclose(response,np.r_[np.zeros(5),np.deg2rad(2)],atol=1e-12)
    assert local_pose_metrics([.1,.2,.3],target,[.1,.2,.3],reference)['local_pose_tracking_met']
    wrong_axis=Rotation.from_euler('x',2,degrees=True).as_matrix()@reference
    assert not local_pose_metrics([.1,.2,.3],wrong_axis,[.1,.2,.3],reference)['local_pose_tracking_met']


def test_bounded_fit_compensates_reference_drift_and_disables_censored_motor():
    matrix=np.zeros((6,16))
    matrix[:,:6]=np.diag([.01,.01,.01,1.,1.,1.])
    matrix[:,8]=np.nan
    target=np.array([-.0002,0,0,-.003,0,np.deg2rad(2)])
    fit=fit_bounded_response(matrix,target)
    assert fit['solver_success'] and fit['rank']==6
    assert fit['censored_columns']==[8] and fit['residual_action'][8]==0
    assert max(np.abs(fit['residual_action']))<=.04
    np.testing.assert_allclose(np.array(fit['predicted_delta'])[:3],target[:3],atol=1e-7)
    assert np.linalg.norm(np.rad2deg(np.array(fit['predicted_delta'])[3:]-target[3:]))<.001


def test_insufficient_local_yaw_authority_does_not_predict_target_or_hide_missing_pairs():
    matrix=np.zeros((6,16))
    matrix[5,0]=.01
    matrix[:,1]=np.nan
    fit=fit_bounded_response(matrix,np.r_[np.zeros(5),np.deg2rad(2)])
    assert fit['rank']==1 and fit['censored_columns']==[1]
    prediction=Rotation.from_rotvec(np.array(fit['predicted_delta'])[3:]).as_matrix()
    assert not local_pose_metrics(np.zeros(3),prediction,np.zeros(3),np.eye(3))['local_pose_tracking_met']
    missing=fit_bounded_response(np.full((6,16),np.nan),np.zeros(6))
    assert not missing['solver_success'] and missing['rank']==0
