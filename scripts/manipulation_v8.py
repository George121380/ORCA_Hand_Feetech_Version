"""Explicit v8 CLI. Legacy checkpoints cannot resume or evaluate here.

Same PPO/data split and full-rotation gates as the legacy CLI, with corrected
wrist physics, braking position commands and actual-hand reset checks.
The legacy deploy_manipulation.py intentionally rejects these checkpoints.
"""
from functools import partial

import manipulation as legacy_cli
from orca_feetech.manipulation_v8 import (TaskConfigV8,ManipulationV8Env,TASK_ID_V8,
                                         load_v8_config,task_contract)


if __name__=='__main__':
    legacy_cli.main(
        train_fn=partial(legacy_cli.train,config_class=TaskConfigV8,env_class=ManipulationV8Env,
                         task_id=TASK_ID_V8,config_loader=load_v8_config,task_contract=task_contract()),
        evaluate_fn=partial(legacy_cli.evaluate,env_class=ManipulationV8Env,config_loader=load_v8_config))
