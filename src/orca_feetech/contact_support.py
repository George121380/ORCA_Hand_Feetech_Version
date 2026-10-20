"""Nominal contact-wrench diagnostics; not force closure or hardware estimates.

Uses the model's pyramidal friction cone and MuJoCo contact-frame convention.
No torques, forces, object states or solver parameters are changed here.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear


def hand_object_contacts(model,data,object_geoms):
    if model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError('Support audit currently requires a pyramidal friction cone')
    contacts=[]
    excluded=[]
    for index,c in enumerate(data.contact):
        if not any(g in (c.geom1,c.geom2) for g in object_geoms) or c.dist>0 or c.efc_address<0:
            continue
        other=c.geom2 if c.geom1 in object_geoms else c.geom1
        body=model.body(model.geom_bodyid[other]).name or 'world'
        if not body.startswith('right_') or body=='right_tower':
            excluded.append(body)
            continue
        local=np.zeros(6)
        mujoco.mj_contactForce(model,data,index,local)
        # Contact-frame forces act from geom1 toward geom2. frame stores its
        # three axes as rows in world coordinates; local X is the normal.
        sign=1. if c.geom2 in object_geoms else -1.
        frame=c.frame.reshape(3,3)
        contacts.append(dict(body=body,position_m=c.pos.copy(),frame=frame.copy(),sign=sign,
            dimension=int(c.dim),friction=c.friction.copy(),penetration_m=float(-c.dist),
            solver_force_world_n=sign*frame.T@local[:3],
            solver_torque_world_nm=sign*frame.T@local[3:]))
    return contacts,excluded


def pyramid_wrenches(contact,center):
    dimension=contact['dimension']
    if dimension not in (1,3,4,6):
        raise ValueError('Unsupported contact dimension')
    frame,sign=contact['frame'],contact['sign']
    normal=np.zeros(6)
    normal[0]=1.
    local=[]
    if dimension==1:
        local.append(normal)
    else:
        for axis in range(1,dimension):
            for direction in (-1.,1.):
                edge=normal.copy()
                edge[axis]=direction*contact['friction'][axis-1]
                local.append(edge)
    columns=[]
    lever=np.asarray(contact['position_m'])-center
    for edge in local:
        force=sign*frame.T@edge[:3]
        torque=sign*frame.T@edge[3:]+np.cross(lever,force)
        columns.append(np.r_[force,torque])
    return np.array(columns).T


def gravity_support(contacts,center,mass,gravity,torque_length=.1):
    """Fit nonnegative pyramid-edge forces to gravity's opposite wrench.

    Coefficients have no actuator/normal-force bound. Feasibility is only for
    this one gravity wrench at this one geometry, not arbitrary disturbances.
    """
    center=np.asarray(center,float)
    gravity=np.asarray(gravity,float)
    weight=float(mass*np.linalg.norm(gravity))
    if (center.shape!=(3,) or gravity.shape!=(3,) or not np.isfinite(center).all()
            or not np.isfinite(gravity).all() or not np.isfinite([mass,torque_length]).all()
            or weight<=0 or torque_length<=0):
        raise ValueError('Expected a finite CoM, positive mass, gravity and torque length')
    target=np.r_[-mass*gravity,np.zeros(3)]
    actual=np.zeros(6)
    for c in contacts:
        force=c['solver_force_world_n']
        actual+=np.r_[force,c['solver_torque_world_nm']+np.cross(c['position_m']-center,force)]
    scale=np.r_[np.ones(3),np.ones(3)/torque_length]
    matrix=np.concatenate([pyramid_wrenches(c,center) for c in contacts],axis=1) if contacts else np.zeros((6,0))
    fitted=np.zeros(6)
    coefficients=np.zeros(matrix.shape[1])
    converged=True
    if matrix.shape[1]:
        fit=lsq_linear(scale[:,None]*matrix,scale*target,bounds=(0,np.inf),
                       method='bvls',max_iter=100,tol=1e-10)
        coefficients=fit.x
        fitted=matrix@coefficients
        converged=bool(fit.success)
    residual=fitted-target
    relative=float(np.linalg.norm(scale*residual)/weight)
    solver_relative=float(np.linalg.norm(scale*(actual-target))/weight)
    normals=[c['sign']*c['frame'][0] for c in contacts]
    opposition=[float(normals[i]@normals[j]) for i in range(len(normals))
                for j in range(i) if contacts[i]['body']!=contacts[j]['body']]
    return dict(contacts=len(contacts),contact_bodies=sorted(set(c['body'] for c in contacts)),
        gravity_magnitude_n=weight,torque_length_m=torque_length,
        cone_fit_converged=converged,gravity_wrench_relative_residual=relative,
        gravity_wrench_feasible_5pct=bool(converged and relative<=.05),
        required_normal_force_sum_n=float(coefficients.sum()),
        fitted_contact_force_world_n=fitted[:3].tolist(),fitted_contact_torque_world_nm=fitted[3:].tolist(),
        solver_contact_force_world_n=actual[:3].tolist(),solver_contact_torque_world_nm=actual[3:].tolist(),
        solver_gravity_wrench_relative_residual=solver_relative,
        minimum_normal_dot_between_bodies=min(opposition,default=None),
        contacts_with_normal_against_gravity=sum(float(n@(-gravity/np.linalg.norm(gravity)))>.2 for n in normals))
