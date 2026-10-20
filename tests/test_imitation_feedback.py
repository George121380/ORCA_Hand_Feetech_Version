from pathlib import Path

import numpy as np
import pytest


def test_feedback_endpoints_are_exact_and_intermediate_actions_remain_bounded(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    from audit_imitation_feedback import FeedbackBlend
    class Policy:
        def __init__(self,action):
            self.action = np.asarray(action,dtype=np.float32)
            self.calls = 0
        def predict(self,obs,deterministic):
            assert deterministic
            self.calls += 1
            return self.action.copy(), None
    source = Policy(np.linspace(-1,1,16))
    candidate = Policy(np.linspace(1,-1,16))
    residual = np.full(16,.35,np.float32)
    actual = FeedbackBlend(source,candidate,residual,0).predict(np.zeros(296))[0]
    np.testing.assert_array_equal(actual,candidate.action)
    assert source.calls == 0
    expected = np.clip(source.action+residual,-1,1)
    np.testing.assert_array_equal(FeedbackBlend(source,candidate,residual,1).predict(None)[0],expected)
    assert candidate.calls == 1
    for fraction in (.25,.5,.75):
        mixed = FeedbackBlend(source,candidate,residual,fraction).predict(None)[0]
        np.testing.assert_allclose(mixed,(1-fraction)*candidate.action+fraction*expected)
        assert np.max(np.abs(mixed)) <= 1 and mixed.dtype == np.float32
    with pytest.raises(ValueError,match="fraction"):
        FeedbackBlend(source,candidate,residual,.1)
    with pytest.raises(ValueError,match="deterministic"):
        FeedbackBlend(source,candidate,residual,0).predict(None,deterministic=False)
