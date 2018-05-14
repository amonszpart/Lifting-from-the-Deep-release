# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:04 2017

@author: Denis Tome'
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.config import JOINT_DRAW_SIZE, LIMB_DRAW_SIZE


def draw_limbs(image, pose_2d, visible, pose_ids=None):
    """Draw the 2D pose without the occluded/not visible joints."""

    _COLORS = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
               [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]
    _LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13]).reshape((-1, 2))

    for oid in xrange(pose_2d.shape[0]):
        if pose_ids is not None and oid not in pose_ids:
            continue
        for lid, (p0, p1) in enumerate(_LIMBS):
            if not (visible[oid][p0] and visible[oid][p1]):
                continue
            y0, x0 = (int(round(c)) for c in pose_2d[oid][p0])
            y1, x1 = (int(round(c)) for c in pose_2d[oid][p1])
            cv2.circle(image, (x0, y0), JOINT_DRAW_SIZE, _COLORS[lid], -1)
            cv2.circle(image, (x1, y1), JOINT_DRAW_SIZE, _COLORS[lid], -1)
            cv2.line(image, (x0, y0), (x1, y1), _COLORS[lid], LIMB_DRAW_SIZE, 16)
            cv2.putText(image, text="%d" % (p0), org=(x0, y0),
                        fontFace=2, fontScale=0.75, color=(200, 100, 100))
            cv2.putText(image, text="%d" % (p1), org=(x1, y1),
                        fontFace=2, fontScale=0.75, color=(200, 100, 100))
        c = int(round(pose_2d[oid][0, 0])), int(round(pose_2d[oid][0, 1]))
        cv2.putText(image, text="%d" % oid,
                    org=(c[1], c[0]), fontFace=1, fontScale=3,
                    color=(255, 200, 255))


def plot_pose(pose):
    """Plot the 3D pose showing the joint connections."""

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                   [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    def joint_color(j):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)


