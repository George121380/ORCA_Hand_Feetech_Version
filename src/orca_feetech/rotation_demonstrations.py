"""Training-only seed allocation and admission; never grants policy acceptance."""


def training_states():
    return [dict(seed=3111000+i,placement_index=i%7,direction=-1 if (i//7)%2==0 else 1)
            for i in range(28)]


def eligible_demo(case):
    return bool(case.get('native_task_success') and case.get('full_uncensored_duration')
                and case.get('command_limits_met') and case.get('end',{}).get('is_success')
                and not case.get('end',{}).get('dropped',True))


def demonstration_admission(cases):
    successful=[c for c in cases if eligible_demo(c)]
    states={(c['seed'],c['placement_index'],c['direction']) for c in successful}
    directions={str(d):sum(s[2]==d for s in states) for d in (-1,1)}
    placements=sorted({s[1] for s in states})
    admitted=len(states)>=8 and min(directions.values())>=2 and len(placements)>=4
    return dict(successful_trajectories=len(successful),successful_states=len(states),
        successful_states_per_direction=directions,successful_placements=placements,
        imitation_admitted=admitted,is_formal_evaluation=False,acceptance_met=False,
        continuous_rotation_accepted=False)
