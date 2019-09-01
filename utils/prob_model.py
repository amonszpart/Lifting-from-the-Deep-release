# -*- coding: utf-8 -*-
"""
Created on Apr 21 13:53 2017

@author: Denis Tome'
"""
from operator import pos

import scipy.io as sio
import numpy as np
from upright_fast import pick_e
import utils.config as config

_H36M_ORDER = [8, 9, 10, 11, 12, 13, 1, 0, 5, 6, 7, 2, 3, 4]
_H36M_REV = [_H36M_ORDER.index(i) for i in range(len(_H36M_ORDER))]

class Prob3dPose:

    def __init__(self, cam_matrix=[]):
        model_param = sio.loadmat('saved_sessions/prob_model/prob_model_params.mat')
        self.mu = np.reshape(model_param['mu'], (model_param['mu'].shape[0], 3, -1))
        self.e = np.reshape(model_param['e'], (model_param['e'].shape[0], model_param['e'].shape[1], 3, -1))
        self.sigma = model_param['sigma']
        self.cam = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    @staticmethod
    def cost3d(model, gt):
        "3d error in mm"
        out = np.sqrt(((gt - model) ** 2).sum(1)).mean(-1)
        return out

    @staticmethod
    def renorm_gt(gt):
        """Compel gt data to have mean joint length of one"""
        _POSE_TREE = np.asarray([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                                [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]).T
        scale = np.sqrt(((gt[:, :, _POSE_TREE[0]] -
                          gt[:, :, _POSE_TREE[1]]) ** 2).sum(2).sum(1))
        return gt / scale[:, np.newaxis, np.newaxis]

    @staticmethod
    def build_model(a, e, s0):
        """Build 3D model"""
        assert (s0.shape[1] == 3)
        assert (e.shape[2] == 3)
        assert (a.shape[1] == e.shape[1])
        # mask = np.where(np.abs(a) > 10000.)
        # if len(mask):
        #     print("[ARON][prob_model.py::build_model] TRUNCATING pose COEFFS")
        #     a[mask] = 1
        out = np.einsum('...i,...ijk', a, e)
        out += s0
        return out

    @staticmethod
    def build_and_rot_model(a, e, s0, r):
        """Build model and rotate according to the identified rotation matrix"""
        from numpy.core.umath_tests import matrix_multiply

        r2 = Prob3dPose.upgrade_r(r.T).transpose((0, 2, 1))
        mod = Prob3dPose.build_model(a, e, s0)
        mod = matrix_multiply(r2, mod)
        return mod

    @staticmethod
    def upgrade_r(r):
        """Upgrades complex parameterisation of planar rotation to tensor containing
        per frame 3x3 rotation matrices"""
        assert (r.ndim == 2)
        # Technically optional assert, but if this fails data is probably transposed
        assert (r.shape[1] == 2)
        assert (np.all(np.isfinite(r)))
        norm = np.sqrt((r[:, :2] ** 2).sum(1))
        assert (np.all(norm > 0))
        r /= norm[:, np.newaxis]
        assert (np.all(np.isfinite(r)))
        newr = np.zeros((r.shape[0], 3, 3))
        newr[:, :2, 0] = r[:, :2]
        newr[:, 2, 2] = 1
        newr[:, 1::-1, 1] = r[:, :2]
        newr[:, 0, 1] *= -1
        return newr

    @staticmethod
    def centre(data_2d):
        """center data according to each of the coordiante components"""
        return (data_2d.T - data_2d.mean(1)).T

    @staticmethod
    def centre_all(data):
        """center all data"""
        try:
            if data.ndim == 2:
                return Prob3dPose.centre(data)
            return (data.transpose(2, 0, 1) - data.mean(2)).transpose(1, 2, 0)
        except ValueError:
            # print("Could not center")
            return data

    @staticmethod
    def normalise_data(d2):
        """Normalise data according to hight"""
        d2 = Prob3dPose.centre_all(d2.reshape(d2.shape[0], -1, 2).transpose(0, 2, 1))

        # Height normalisation (2 meters)
        m2 = d2[:, 1, :].min(1) / 2.0
        m2 -= d2[:, 1, :].max(1) / 2.0
        crap = m2 == 0
        m2[crap] = 1.0
        d2 /= m2[:, np.newaxis, np.newaxis]
        return d2, m2

    @classmethod
    def transform_joints(cls, pose_2d, visible_joints):
        """Transform the set of joints according to what the probabilistic model expects as input.
        It returns the new set of joints of each of the people and the set of weights for the joints."""

        # _H36M_ORDER = [8, 9, 10, 11, 12, 13, 1, 0, 5, 6, 7, 2, 3, 4]
        _H36M_ORDER = cls._H36M_ORDER
        _W_POS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

        def swap_xy(poses):
            tmp = np.copy(poses[:, :, 0])
            poses[:, :, 0] = poses[:, :, 1]
            poses[:, :, 1] = tmp
            return poses

        assert (pose_2d.ndim == 3)
        new_pose = pose_2d.copy()
        new_pose = swap_xy(new_pose)
        new_pose = new_pose[:, _H36M_ORDER]

        # defining weights according to occlusions
        weights = np.zeros((pose_2d.shape[0], 2, config.H36M_NUM_JOINTS))
        ordered_visibility = \
            np.repeat(visible_joints[:, _H36M_ORDER, np.newaxis], 2, 2) \
                .transpose([0, 2, 1])
        weights[:, :, _W_POS] = ordered_visibility
        return new_pose, weights

    def affine_estimate(self, w, depth_reg=0.085, weights=np.zeros((0, 0, 0)),
                        scale=10.0, scale_mean=0.0016 * 1.8 * 1.2,
                        scale_std=1.2 * 0, cap_scale=-0.00129):
        """Quick switch to allow reconstruction at unknown scale
        returns a,r and scale
        """
        s = np.empty((self.sigma.shape[0], self.sigma.shape[1] + 4))  # e,y,x,z
        s[:, :4] = 10 ** -5  # Tiny but makes stuff well-posed
        s[:, 0] = scale_std
        s[:, 4:] = self.sigma
        s[:, 4:-1] *= scale

        e2 = np.zeros((self.e.shape[0], self.e.shape[1] + 4, 3, self.e.shape[3]))
        e2[:, 1, 0] = 1.0
        e2[:, 2, 1] = 1.0
        e2[:, 3, 0] = 1.0
        # This makes the least_squares problem ill posed,
        # as X,Z are interchangeable
        # Hence regularisation above to speed convergence and stop blow-up
        e2[:, 0] = self.mu
        e2[:, 4:] = self.e
        t_m = np.zeros_like(self.mu)
        res, a, r = pick_e(w, e2, t_m, self.cam, s, weights=weights,
                           interval=0.01, depth_reg=depth_reg,
                           scale_prior=scale_mean)
        # print("pick_e done")
        scale = a[:, :, 0]
        reestimate = scale > cap_scale
        m = self.mu * cap_scale
        for i in xrange(scale.shape[0]):
            if reestimate[i].sum() > 0:
                ehat = e2[i:i + 1, 1:]
                mhat = m[i:i + 1]
                shat = s[i:i + 1, 1:]
                (res2, a2, r2) = pick_e(w[reestimate[i]], ehat, mhat, self.cam, shat,
                                        weights=weights[reestimate[i]],
                                        interval=0.01, depth_reg=depth_reg,
                                        scale_prior=scale_mean)
                res[i:i + 1, reestimate[i]] = res2
                a[i:i + 1, reestimate[i], 1:] = a2
                a[i:i + 1, reestimate[i], 0] = cap_scale
                r[i:i + 1, :, reestimate[i]] = r2
        scale = a[:, :, 0]
        a = a[:, :, 1:] / a[:, :, 0][:, :, np.newaxis]
        return res, e2[:, 1:], a, r, scale

    def better_rec(self, w, model, s=1, weights=1, damp_z=1):
        """Quick switch to allow reconstruction at unknown scale
        returns a,r and scale"""
        from numpy.core.umath_tests import matrix_multiply
        proj = matrix_multiply(self.cam[np.newaxis], model)
        proj[:, :2] = (proj[:, :2] * s + w * weights) / (s + weights)
        proj[:, 2] *= damp_z
        out = matrix_multiply(self.cam.T[np.newaxis], proj)
        return out

    def create_rec(self, w2, weights, res_weight=1):
        """Reconstruct 3D pose given a 2D pose"""
        _SIGMA_SCALING = 5.2

        # print("calling affine_estimate on w2: %s" % repr(w2.shape))
        res, e, a, r, scale = \
            self.affine_estimate(
                w2, scale=_SIGMA_SCALING, weights=weights,
                depth_reg=0, cap_scale=-0.001, scale_mean=-0.003)
        # print("res: %s" % repr(res))
        remaining_dims = 3 * w2.shape[2] - e.shape[1]
        assert (remaining_dims >= 0)
        llambda = -np.log(self.sigma)
        lgdet = np.sum(llambda[:, :-1], 1) + llambda[:, -1] * remaining_dims
        score = (res * res_weight + lgdet[:, np.newaxis] * (scale ** 2))
        best = np.argmin(score, 0)
        index = np.arange(best.shape[0])
        a2 = a[best, index]
        r2 = r[best, :, index].T
        rec = Prob3dPose.build_and_rot_model(a2, e[best], self.mu[best], r2)
        rec *= -np.abs(scale[best, index])[:, np.newaxis, np.newaxis]

        rec = self.better_rec(w2, rec, 1, 1.55 * weights, 1) * -1
        rec = Prob3dPose.renorm_gt(rec)
        rec *= 0.97
        return rec

    def compute_3d(self, pose_2d, weights, is_openpose=False):
        """Reconstruct 3D poses given 2D estimations"""

        _J_POS =[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16] # original
        # if is_openpose:
        #     # OpenPose returns neck (ie. nose) and not tip of head, so 8 => 9
        #     _J_POS = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
        #     weights[:, :, 9] = weights[:, :, 8]
        #     weights[:, :, 8] = 0
        _SCALE_3D = 1174.88312988

        if pose_2d.shape[1] != config.H36M_NUM_JOINTS:
            # need to call the linear regressor
            reg_joints = np.zeros((pose_2d.shape[0], config.H36M_NUM_JOINTS, 2))
            for oid, singe_pose in enumerate(pose_2d):
                reg_joints[oid, _J_POS] = singe_pose

            norm_pose, _ = Prob3dPose.normalise_data(reg_joints)
            # print("normalise_data finished (branch 0)")
        else:
            norm_pose, _ = Prob3dPose.normalise_data(pose_2d)
            # print("normalise_data finished (branch 1)")

        pose_3d = self.create_rec(norm_pose, weights) * _SCALE_3D
        # print("create_rec finished")
        return pose_3d
