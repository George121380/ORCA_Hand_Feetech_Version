import mujoco
import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv,TaskConfig
from orca_feetech.wrist_diagnostic import NeutralWristDiagnosticEnv


def test_engine_joint_equality_constant_is_relative_to_joint_reference():
    for constant,expected in ((-.15,-.30),(0.,-.15)):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
          <option gravity="0 0 0" timestep=".002"/>
          <worldbody><body><joint name="j" type="hinge" ref="-.15"/>
          <geom type="sphere" size=".02" mass=".1"/></body></worldbody>
          <equality><joint joint1="j" polycoef="{constant} 0 0 0 0"/></equality></mujoco>''')
        data=mujoco.MjData(model)
        mujoco.mj_step(model,data,nstep=1000)
        assert model.qpos0[0]==pytest.approx(-.15)
        assert data.qpos[0]==pytest.approx(expected,abs=1e-6)


def test_corrected_diagnostic_wrist_tracks_neutral_and_cannot_silently_train():
    legacy=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    corrected=NeutralWristDiagnosticEnv(TaskConfig(object='pen',randomization=0))
    try:
        assert legacy.model.eq_data[0,0]==pytest.approx(-.15)
        assert corrected.model.eq_data[0,0]==pytest.approx(0.)
        for env in (legacy,corrected):
            env.data.qpos[env.qadr]=env.hand_spec.neutral
            env.data.ctrl[env.actuators]=env.hand_spec.neutral
            env.data.qpos[env.oq:env.oq+7]=[.6,0,.8,1,0,0,0]
            mujoco.mj_forward(env.model,env.data)
            mujoco.mj_step(env.model,env.data,nstep=2500)
        assert abs(legacy.data.qpos[legacy.qadr[-1]]+.15)>.1
        assert abs(corrected.data.qpos[corrected.qadr[-1]]+.15)<.01
        assert corrected.wrist_contract()['equality_coefficients'][0]==pytest.approx(0.)
        with pytest.raises(RuntimeError,match='new registered task contract'):
            corrected.step(np.zeros(16))
        # Constructing the candidate must not mutate another v7 model.
        assert legacy.model.eq_data[0,0]==pytest.approx(-.15)
    finally:
        legacy.close()
        corrected.close()
