"""Deployable deterministic features computed only from the original raw input."""
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RelativeGoalExtractor(BaseFeaturesExtractor):
    """Replace absolute G by G @ R.T in each74-value history frame.

    R and G are already present in the296 raw observations. Retaining R makes
    the transform invertible for rotations: G = (G @ R.T) @ R. No simulation
    truth, contact, time or extra observation is introduced.
    """
    def __init__(self,observation_space):
        if observation_space.shape!=(296,):
            raise ValueError('Relative goal features require raw4x74 observations')
        super().__init__(observation_space,features_dim=296)

    def forward(self,observations):
        frames=observations.reshape(-1,4,74)
        rotation=frames[:,:,38:47].reshape(-1,4,3,3)
        goal=frames[:,:,47:56].reshape(-1,4,3,3)
        result=frames.clone()
        result[:,:,47:56]=(goal@rotation.transpose(-1,-2)).reshape(-1,4,9)
        return result.reshape(-1,296)
