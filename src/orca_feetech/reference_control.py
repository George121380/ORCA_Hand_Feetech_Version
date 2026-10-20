"""Candidate command contract for future reference tasks; legacy v7 is unchanged.

Positions use q_next = q + velocity_next * DT. A discrete stopping-distance
constraint reserves enough ROM for all subsequent deceleration steps. This
limits effective position-command acceleration, including at ROM boundaries.
It does not bound measured actuator acceleration or validate real dynamics.
"""
import numpy as np

from .manipulation import DT, limits
from .model import ModelSpec, vector

CONTROLLER_ID = 'orca-discrete-braking-v1'


def stoppable_speed(distance,acceleration,dt=DT):
    """Invert dt * sum(max(speed - k*acceleration*dt, 0)), including this step."""
    distance=np.maximum(np.asarray(distance,float),0.)
    decrement=acceleration*dt
    # On the interval ((n-1)*decrement, n*decrement], stopping distance is
    # dt * (n*speed - decrement*n*(n-1)/2). Its inverse is piecewise linear.
    n=np.maximum(1.,np.ceil((np.sqrt(1+8*distance/(dt*decrement))-1)/2))
    return (distance/dt+decrement*n*(n-1)/2)/n


class BrakingActionCodec:
    """Separate opt-in contract, not a drop-in migration for trained v7 actors.

    Invalid object observations request bounded deceleration rather than an
    instantaneous position hold. No training/deployment entrypoint selects
    this codec yet. The future task must version that loss-of-vision behavior.
    """
    controller_id=CONTROLLER_ID

    def __init__(self,config,initial):
        self.config=config
        self.lower,self.upper=limits(config)
        self.q=vector(initial)
        self.wrist=ModelSpec.load().neutral[-1]
        if (np.any(self.q<self.lower) or np.any(self.q>self.upper)
                or abs(self.q[-1]-self.wrist)>1e-10):
            raise ValueError('Initial command must be within ROM with neutral wrist')
        self.velocity=np.zeros(16)

    def step(self,action,valid=True):
        action=np.asarray(action,float)
        if action.shape!=(16,) or not np.isfinite(action).all():
            raise ValueError('Expected 16 finite actions')
        acceleration=self.config.max_acceleration
        delta_velocity=acceleration*DT
        speed=self.config.max_velocity
        safe_negative=stoppable_speed(self.q[:16]-self.lower[:16],acceleration)
        safe_positive=stoppable_speed(self.upper[:16]-self.q[:16],acceleration)
        low=np.maximum.reduce([self.velocity-delta_velocity,np.full(16,-speed),-safe_negative])
        high=np.minimum.reduce([self.velocity+delta_velocity,np.full(16,speed),safe_positive])
        if np.any(low>high+1e-10):
            raise RuntimeError('Controller state outside discrete braking feasibility set')
        # Collapse only a floating-point-sized overlap inversion.
        middle=(low+high)/2
        invalid=low>high
        low=np.where(invalid,middle,low)
        high=np.where(invalid,middle,high)
        requested=speed*np.clip(action,-1,1) if valid else np.zeros(16)
        next_velocity=np.clip(requested,low,high)
        next_q=self.q[:16]+next_velocity*DT
        if np.any(next_q<self.lower[:16]-1e-10) or np.any(next_q>self.upper[:16]+1e-10):
            raise RuntimeError('Discrete braking produced a position outside ROM')
        # Clean up roundoff only; substantial position clipping is forbidden.
        next_q=np.clip(next_q,self.lower[:16],self.upper[:16])
        self.velocity=(next_q-self.q[:16])/DT
        self.q[:16]=next_q
        self.q[-1]=self.wrist
        return self.q.copy()
