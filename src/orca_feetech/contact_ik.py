"""Contact-point IK proposals on isolated kinematic data, not live control.

No rigid grasp, friction or force-closure assumption is imposed on execution.
The original dynamics must separately validate every proposed command path.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares


def preload_bounds(actual,command,lower,upper,limit=.3):
    actual,command,lower,upper=map(lambda a:np.asarray(a,dtype=float),(actual,command,lower,upper))
    low=np.maximum.reduce([actual-limit,lower,actual+lower-command])
    high=np.minimum.reduce([actual+limit,upper,actual+upper-command])
    if (not np.isfinite([actual,command,lower,upper]).all() or np.any(low>=high)
            or np.any(command<lower) or np.any(command>upper)):
        raise ValueError('Initial actual joints/preloaded commands outside fixed IK bounds')
    return low,high


def transformed_anchors(anchors,center,delta_rotation,translation):
    center,delta_rotation,translation=map(np.asarray,(center,delta_rotation,translation))
    points=np.array([center+translation+delta_rotation@(np.asarray(a['world_point'])-center) for a in anchors])
    normals=np.array([delta_rotation@np.asarray(a['world_normal']) for a in anchors])
    return points,normals


class ContactKinematics:
    def __init__(self,model,source,qadr,anchors,command,lower,upper):
        self.model=model
        self.data=mujoco.MjData(model)
        self.data.qpos[:]=source.qpos
        self.data.qvel[:]=source.qvel
        self.qadr=np.asarray(qadr,dtype=int)
        self.initial=source.qpos[self.qadr].copy()
        self.command=np.asarray(command).copy()
        self.lower,self.upper=preload_bounds(self.initial,self.command,lower,upper)
        self.anchors=anchors

    def attached(self,q):
        self.data.qpos[self.qadr]=q
        mujoco.mj_kinematics(self.model,self.data)
        points=[]
        normals=[]
        for anchor in self.anchors:
            body=anchor['body_id']
            rotation=self.data.xmat[body].reshape(3,3)
            points.append(self.data.xpos[body]+rotation@np.asarray(anchor['local_point']))
            normals.append(rotation@np.asarray(anchor['local_normal']))
        return np.asarray(points),np.asarray(normals)

    def fit(self,target_points,target_normals,previous,*,normal_targets):
        target_points,target_normals,previous=map(np.asarray,(target_points,target_normals,previous))
        def residual(q):
            points,normals=self.attached(q)
            terms=[((points-target_points)/.001).ravel(),.05*(q-self.initial)/.3,.05*(q-previous)/.3]
            if normal_targets:
                terms.append(((normals-target_normals)/np.deg2rad(5.)).ravel())
            return np.concatenate(terms)
        result=least_squares(residual,np.clip(previous,self.lower,self.upper),bounds=(self.lower,self.upper),
            max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        points,normals=self.attached(result.x)
        normal_errors=np.rad2deg(np.arccos(np.clip(np.sum(normals*target_normals,axis=1),-1,1)))
        command=self.command+result.x-self.initial
        return dict(actual_joint_target=result.x.tolist(),command_target=command.tolist(),
            success=bool(result.success),nfev=int(result.nfev),cost=float(result.cost),
            target_points=np.asarray(target_points).tolist(),target_normals=np.asarray(target_normals).tolist(),
            actual_points=points.tolist(),actual_normals=normals.tolist(),
            point_errors_m=np.linalg.norm(points-target_points,axis=1).tolist(),normal_errors_deg=normal_errors.tolist())


def contact_command(initial,knots,seconds):
    """Four fixed1.25s quintic segments, then unchanged target to7s."""
    initial=np.asarray(initial,dtype=float)
    knots=np.asarray(knots,dtype=float)
    if (knots.shape!=(4,len(initial)) or not np.isfinite(knots).all()
            or not np.isfinite(seconds) or not 0<=seconds<=7.+1e-10):
        raise ValueError('Expected four finite command knots within the7s horizon')
    segment=min(int(seconds/1.25),3)
    start=initial if segment==0 else knots[segment-1]
    end=knots[segment]
    phase=float(np.clip((seconds-segment*1.25)/1.25,0,1))
    blend=phase**3*(10+phase*(-15+6*phase))
    return start+blend*(end-start)
