"""Policy-local features computed solely from the deployable v7 observation."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .manipulation import FRAME_SIZE, HISTORY


class GoalErrorFeatures(BaseFeaturesExtractor):
    """Expose G R^T explicitly; the camera/encoder observation stays unchanged."""
    def __init__(self, observation_space):
        if FRAME_SIZE != 74 or observation_space.shape != (HISTORY * FRAME_SIZE,):
            raise ValueError("Goal-error features require the v7 74-by-4 observation contract")
        super().__init__(observation_space, features_dim=HISTORY * (FRAME_SIZE + 9))

    def forward(self, observations):
        frames = observations.reshape(-1, HISTORY, FRAME_SIZE)
        rotation = frames[..., 38:47].reshape(-1, HISTORY, 3, 3)
        goal = frames[..., 47:56].reshape(-1, HISTORY, 3, 3)
        relative = goal @ rotation.transpose(-2, -1)
        return torch.cat((frames, relative.reshape(-1, HISTORY, 9)), dim=-1).flatten(1)


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
