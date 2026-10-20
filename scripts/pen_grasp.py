"""Explicit seven-second loaded-grasp training/evaluation; never rotation credit."""
from functools import partial

import manipulation as cli
from orca_feetech.pen_grasp import (GraspConfig,PenGraspEnv,TASK_ID_GRASP,
                                   load_grasp_config,grasp_contract)


if __name__=='__main__':
    cli.main(
        train_fn=partial(cli.train,config_class=GraspConfig,env_class=PenGraspEnv,
                         task_id=TASK_ID_GRASP,config_loader=load_grasp_config,task_contract=grasp_contract()),
        evaluate_fn=partial(cli.evaluate,env_class=PenGraspEnv,config_loader=load_grasp_config,
                            reset_options=lambda episode: {'placement_index':episode%7}),
        train_defaults=dict(object='pen',angle=10,horizon=140,randomization=0.,initial_log_std=-2.))
