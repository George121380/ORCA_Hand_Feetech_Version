import gymnasium as gym
import numpy as np
import pytest
import torch

from orca_feetech.manipulation import yaw_rotation
from orca_feetech.policy_features import GoalErrorFeatures


def test_goal_features_use_observed_pose_and_are_invariant_to_common_body_rotation():
    extractor = GoalErrorFeatures(gym.spaces.Box(-np.inf, np.inf, (296,), np.float32))
    rng = np.random.default_rng(29)
    inputs = rng.normal(size=(2, 4, 74)).astype(np.float32)
    for sample in range(2):
        for frame in range(4):
            R = yaw_rotation(.2*frame + sample)
            G = yaw_rotation(.6) @ R
            inputs[sample, frame, 38:47] = R.ravel()
            inputs[sample, frame, 47:56] = G.ravel()
    transformed = inputs.copy()
    S = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    for start in (38, 47):
        transformed[..., start:start+9] = (inputs[..., start:start+9].reshape(2, 4, 3, 3) @ S).reshape(2, 4, 9)
    a = extractor(torch.from_numpy(inputs.reshape(2, 296))).reshape(2, 4, 83)
    b = extractor(torch.from_numpy(transformed.reshape(2, 296))).reshape(2, 4, 83)
    assert torch.equal(a[..., :74], torch.from_numpy(inputs))
    assert torch.allclose(a[..., 74:], b[..., 74:], atol=1e-6)
    assert np.allclose(a[0, 0, 74:].numpy().reshape(3, 3), yaw_rotation(.6), atol=1e-6)
    changed = inputs.copy()
    changed[0, 0, 38:47] = yaw_rotation(.3).ravel()
    c = extractor(torch.from_numpy(changed.reshape(2, 296))).reshape(2, 4, 83)
    assert not torch.allclose(a[0, 0, 74:], c[0, 0, 74:])
    # Each history frame must use its own held/noisy camera pose, without any
    # simulator state or an extra source of fresh observations.
    assert torch.equal(a[0, 1:], c[0, 1:])
    with pytest.raises(ValueError, match="v7"):
        GoalErrorFeatures(gym.spaces.Box(-np.inf, np.inf, (232,), np.float32))
