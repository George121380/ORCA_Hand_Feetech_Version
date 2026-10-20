"""Isolated wrist-reference correction, intentionally unavailable to RL entrypoints.

Legacy v7 stays reproducible. Register a new task/checkpoint/controller contract
before using this model for policy training or deployment.
"""
import hashlib
import json

import mujoco
import numpy as np

from .manipulation import ManipulationEnv


MODEL_CONTRACT='orca-v1-neutral-wrist-diagnostic-v1'


class NeutralWristDiagnosticEnv(ManipulationEnv):
    diagnostic_model_contract=MODEL_CONTRACT

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        wrist=self.model.joint('right_wrist').id
        locks=[i for i,t in enumerate(self.model.eq_type) if t==mujoco.mjtEq.mjEQ_JOINT
               and self.model.eq_obj1id[i]==wrist and self.model.eq_obj2id[i]==-1]
        if len(locks)!=1:
            raise ValueError('Expected exactly one scalar wrist equality')
        self.wrist_equality_id=locks[0]
        reference=float(self.model.qpos0[self.qadr[-1]])
        # MuJoCo: q - q_reference = polynomial constant when joint2 is absent.
        self.model.eq_data[locks[0],0]=self.hand_spec.neutral[-1]-reference

    def step(self,action):
        raise RuntimeError('Diagnostic wrist model requires a new registered task contract before RL or deployment')

    def wrist_contract(self):
        row=self.model.eq_data[self.wrist_equality_id]
        if np.count_nonzero(row[1:5]):
            raise ValueError('Expected a constant wrist equality')
        payload=dict(model_contract=MODEL_CONTRACT,base_fingerprint=self.config.fingerprint(),
            neutral_wrist_rad=float(self.hand_spec.neutral[-1]),reference_rad=float(self.model.qpos0[self.qadr[-1]]),
            equality_coefficients=row[:5].tolist())
        return dict(**payload,signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
