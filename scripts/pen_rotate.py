"""One retained pen yaw target after a physically executed frozen grasp."""
from functools import partial

import manipulation as cli
from orca_feetech.pen_rotate import (RotationConfig,PenRotateEnv,TASK_ID_ROTATE,
                                    load_rotation_config,rotation_contract)


if __name__=='__main__':
    cli.main(
        train_fn=partial(cli.train,config_class=RotationConfig,env_class=PenRotateEnv,
                         task_id=TASK_ID_ROTATE,config_loader=load_rotation_config,task_contract=rotation_contract()),
        evaluate_fn=partial(cli.evaluate,env_class=PenRotateEnv,config_loader=load_rotation_config,
                            reset_options=lambda episode: {'placement_index':episode%7,
                                                           'direction':-1 if (episode//7)%2==0 else 1}),
        train_defaults=dict(object='pen',angle=15,horizon=140,randomization=.3,initial_log_std=-4.))
