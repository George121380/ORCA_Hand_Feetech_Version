"""Optional fresh-policy actor initialization; learned checkpoints are never reset."""
import torch


def zero_actor_mean(policy):
    """Leave features, critic, exploration and optimizer references intact."""
    if not isinstance(policy.action_net,torch.nn.Linear):
        raise ValueError('Zero mean initialization requires a linear Gaussian actor head')
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
