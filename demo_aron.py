#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'


The function that is doing all the work of computing the 3D poses and trying
to minimise the re-projection error for each of the PPCA models independently
is called pick_e.
This is defined inside /utils/externals/upright_fast.pyx (cython)

If you look at the affine_estimate function, that is where the function is
called and returns among other parameters also a res value: one for each of
the PPCA models.
In the create_rec function you can see how that res value is used to chose
among the 3 PPCA models which is the best.

score = (res * res_weight + lgdet[:, np.newaxis] * (scale ** 2))
best = np.argmin(score, 0)

best contains the index of the best PPCA model to use.
"""
import json
import os
import sys

import skimage  # by Aron for show_heatmaps
import tensorflow as tf

import utils.config as config
import utils.process as ut
from openpose import JointOpenPose
from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.logic.joints import Joint
from imapper.pose.skeleton import JointDenis
from utils import cpm
from utils.draw import *
from utils.prob_model import Prob3dPose

if not sys.version_info[0] < 3:
    from typing import Union

from mpl_toolkits.mplot3d import Axes3D

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest="image", help="Image to work with")
parser.add_argument('-d', dest="image_dir", help="Folder of images to work with")
parser.add_argument('--no-vis', action="store_true", help="No visualization")
parser.add_argument('--vis-thresh', type=float,
                    help='Heatmap detection threshold. Default=1e-3.',
                    default=config.VISIBLE_PART)
parser.add_argument('--thresh-min-max', type=float,
                    help='Heatmap maximum and minimum difference threshold '
                         'that removes spurious 2D pose centroids.',
                    default=0.3)
parser.add_argument('--center-thresh', type=float,
                    help="Heatmap response at pose centroid. Default: 0.4",
                    default=config.CENTER_TR)
parser.add_argument('--dest-dir', type=str, help="Where to save output.",
                    default='denis')
parser.add_argument('--skel2d', type=str, help="2D skeleton file to use.")
parser.add_argument('-start', type=int, help='First frame')
parser.add_argument('-end', type=int, help='Last frame (inclusive)')
args = parser.parse_args()
assert ((args.image is None) != (args.image_dir is None)), \
    "Need either image or directory"


def default_encode(o):
    """JSON encoder for float values"""
    if isinstance(o, float) or isinstance(o, np.float32):
        return "%f" % o
    else:
        print("type: %s" % type(o))
    raise TypeError(repr(o) + " is not JSON serializable")


def show_heatmaps(heatmaps, centers, size, num_parts=14):
    parts = np.zeros((len(centers), num_parts, 2), dtype=np.int32)
    visible = np.zeros((len(centers), num_parts), dtype=bool)
    for oid, (yc, xc) in enumerate(centers):
        part_hmap = skimage.transform.resize(np.clip(heatmaps[oid], -1, 1), size)
        fig, ax = plt.subplots(4, 4)
        for pid in xrange(num_parts):
            ax[np.unravel_index(pid, ax.shape)].imshow(part_hmap[:, :, pid])
            # y, x = np.unravel_index(np.argmax(part_hmap[:, :, pid]), size)
            # parts[oid, pid] = y + yc - size[0] // 2, x + xc - size[1] // 2
            # visible[oid, pid] = np.mean(part_hmap[:, :, pid]) > config.VISIBLE_PART

    plt.show()


def load_image(fname, return_scale=False):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    scale = config.INPUT_SIZE/(image.shape[0] * 1.0)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # print("image.size: %s" % repr(image.shape))
    b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)
    if return_scale:
        return b_image, image, scale
    else:
        return b_image, image

inputs = []
if args.image_dir is not None:
    for f in os.listdir(args.image_dir):
        if f.endswith('jpg') or f.endswith('png'):
            inputs.append(os.path.join(args.image_dir, f))
else:
    fname = args.image or 'images/test_image.png'
    inputs.append(fname)

b_image, image, scale = load_image(inputs[0], return_scale=True)
shape = image.shape

tf.reset_default_graph()

with tf.variable_scope('CPM'):
    # placeholders for person network
    image_in = tf.placeholder(tf.float32, [1, config.INPUT_SIZE, shape[1], 3])
    heatmap_person = cpm.inference_person(image_in)
    heatmap_person_large = \
        tf.image.resize_images(heatmap_person, [config.INPUT_SIZE, shape[1]])

    # placeholders for pose network
    N = 16
    pose_image_in = \
        tf.placeholder(tf.float32, [N, config.INPUT_SIZE, config.INPUT_SIZE, 3])
    pose_centermap_in = \
        tf.placeholder(tf.float32, [N, config.INPUT_SIZE, config.INPUT_SIZE, 1])
    heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

pose_ids = None # []

entries = {"image_shape": shape, 'vis_thresh': args.vis_thresh}
out_dir = None
prev_entry = None
scene_root = os.path.join(os.path.dirname(inputs[0]), os.pardir)

if hasattr(args, 'skel2d') and args.skel2d is not None:
    if not args.skel2d.startswith(os.sep) \
      and hasattr(args, 'image_dir') and args.image_dir is not None:
        args.skel2d = os.path.normpath(
          os.path.join(args.image_dir, os.pardir, args.skel2d))
    args.dest_dir = os.path.join(scene_root, 'gt')
else:
    args.skel2d = None

skel2d = Scenelet.load(args.skel2d).skeleton \
    if args.skel2d is not None \
    else None  # type: Union[Skeleton, None]
if skel2d is not None and skel2d.n_actors > 1:
    skel3d = Skeleton(frames_mod=skel2d.frames_mod, n_actors=skel2d.n_actors,
                      min_frame_id=skel2d.min_frame_id)
else:
    skel3d = Skeleton()
#"build/examples/openpose/openpose.bin --image_dir /media/data/amonszpa/stealth/shared/video_recordings/angrymen00/origjpg/ --write_json /media/data/amonszpa/stealth/shared/video_recordings/angrymen00/openpose_keypoints --display 0 -face"
# -d /media/data/amonszpa/stealth/shared/video_recordings/library1-lcrnet/origjpg --skel2d quant/skel_GT_2d.json --no-vis
for fname_id, fname in enumerate(sorted(inputs)):
    if out_dir is None:
        out_dir = os.path.join(scene_root, args.dest_dir)
        # print("will write to %s" % out_dir)
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
    name_im = os.path.splitext(os.path.split(fname)[1])[0]
    frame_id = int(name_im.split('_')[-1])
    if hasattr(args, 'start') and args.start is not None \
      and frame_id < args.start:
        continue
    elif hasattr(args, 'end') and args.end is not None \
        and args.end < frame_id:
        continue
    p_keypoints = "%s/openpose_keypoints/color_%05d_keypoints.json" \
                  % (scene_root, frame_id)
    # todo: frame_id needs to be created
    is_openpose = False
    if args.skel2d is not None:
        parts = np.array([
            skel2d.get_pose(frame_id=skel2d.unmod_frame_id(
              frame_id=frame_id, actor_id=actor_id,
              frames_mod=skel2d.frames_mod))
            for actor_id in range(skel2d.n_actors)])

        rev_map = [Joint.from_string(JointOpenPose(j).get_name()) for j in range(JointOpenPose.END)]
        parts = np.transpose(
          np.concatenate((parts[:, 1:2, rev_map],
                          parts[:, 0:1, rev_map]), axis=1), axes=(0, 2, 1))
        visible = np.ones(shape=(parts.shape[0], parts.shape[1]))
    elif os.path.exists(p_keypoints):
        is_openpose = True
        data = json.load(open(p_keypoints, 'r'))
        parts, visible, visible_float = \
            JointOpenPose.parse_keypoints(data['people'])
        if False:
            parts = np.concatenate((parts[:, Prob3dPose._H36M_REV, 1:2],
                                    parts[:, Prob3dPose._H36M_REV, 0:1]), axis=2)
            visible = visible[:, Prob3dPose._H36M_REV]
            visible_float = visible_float[:, Prob3dPose._H36M_REV]
        else:
            parts = np.concatenate((parts[:, :, 1:2],
                                    parts[:, :, 0:1]), axis=2)
            visible = visible[:, :]
            visible_float = visible_float[:, :]

        # y, x
        parts, visible, visible_float = \
            JointOpenPose.hallucinate(parts, visible, visible_float)
        # parts[0, JointOpenPose.RANK, :] = (481, 699)
        # parts[0, JointOpenPose.LANK, :] = (498.5, 757)
        # visible[0, JointOpenPose.RANK] = True
        # visible_float[0, JointOpenPose.RANK] = 1.
        # visible[0, JointOpenPose.LANK] = True
        # visible_float[0, JointOpenPose.LANK] = 1.
        # parts[0, JointOpenPose.RANK, :] = (699, 481)
        parts *= scale
    else:

        b_image, image = load_image(fname)
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, 'saved_sessions/person_MPI/init')
            hmap_person = sess.run(heatmap_person_large, {image_in: b_image})

        hmap_person = np.squeeze(hmap_person)
        centers = ut.detect_objects_heatmap(hmap_person,
                                            args.center_thresh,
                                            args.thresh_min_max)
        # print("hmap_person.shape: %s" % repr(hmap_person.shape))
        b_pose_image, b_pose_cmap = \
            ut.prepare_input_posenet(b_image[0], centers,
                                     [config.INPUT_SIZE, shape[1]],
                                     [config.INPUT_SIZE, config.INPUT_SIZE])
        # print("b_pose_image.shape: %s" % repr(b_pose_image.shape))
        # print("b_bose_cmap.shape: %s" % repr(b_pose_cmap.shape))

        sess = tf.InteractiveSession()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, 'saved_sessions/pose_MPI/init')

            feed_dict = {
                pose_image_in: b_pose_image,

                pose_centermap_in: b_pose_cmap
            }

            _hmap_pose = sess.run(heatmap_pose, feed_dict)

        # Estimate 2D poses
        parts, visible, visible_float = \
            ut.detect_parts_heatmaps(_hmap_pose, centers,
                                     [config.INPUT_SIZE, config.INPUT_SIZE],
                                     visible_part_threshold=args.vis_thresh)
    # show_heatmaps(_hmap_pose, centers,
    #               [config.INPUT_SIZE, config.INPUT_SIZE])

    # print("parts:\n%s" % (parts / scale))
    # p_txt = os.path.join(out_dir, os.pardir, "debug3", "parts.txt")
    # np.savetxt(p_txt, parts.reshape((-1, 2)), fmt="%g", delimiter=',')
    try:
    # if True:
        # Estimate 3D poses
        poseLifting = Prob3dPose()
        # parts: (k, 14, 2) int32, where k is the number of skeletons
        # visible: (k, 14) bool
        pose2D, weights = Prob3dPose.transform_joints(parts, visible)
        # print("pose2D: %s" % repr(pose2D.shape))
        # print("ok1")
        pose3D = poseLifting.compute_3d(pose2D, weights,
                                        is_openpose=is_openpose)
        # print("pose3d: %s" % pose3D)
        # print("weights: %s" % weights)
        # print("pose3d.shape: %s" % repr(pose3D.shape))
        # print("weights.shape: %s" % repr(weights.shape))

        if not args.no_vis:
            # Show 2D poses
            plt.figure()
            draw_limbs(image, parts, visible, pose_ids=pose_ids)
            plt.imshow(image)
            p_tmp_dir = os.path.join(out_dir, os.pardir, "debug3")
            try:
                os.makedirs(p_tmp_dir)
            except OSError:
                pass
            p_tmp = os.path.join(p_tmp_dir, "debug_%05d.jpg" % frame_id)
            plt.savefig(p_tmp)
            plt.axis('off')

            # Show 3D poses
            for pid, single_3D in enumerate(pose3D):
                plot_pose(poseLifting.centre_all(single_3D))
                p_tmp = os.path.join(out_dir, os.pardir, "debug3",
                                     "debug_%05d_p%d.jpg" % (frame_id, pid))
                plt.savefig(p_tmp)

            # plt.show()
            plt.close()

        entry = {'pose_2d': parts.tolist(),
                 'visible': visible.astype(int).tolist(),
                 'pose_3d': pose3D.tolist(),
                 'centered_3d': [poseLifting.centre_all(single_3D).tolist()
                                 for single_3D in pose3D]}
        if 'visible_float' in locals():
            entry['visible_float'] = visible_float.tolist()  # "confidence"
        entries[name_im] = entry

        for actor_id, pose in enumerate(entry['centered_3d']):
            frame_id2 = skel3d.unmod_frame_id(frame_id=frame_id,
                                              actor_id=actor_id,
                                              frames_mod=skel3d.frames_mod)
            pose = np.array(pose)
            pose = pose[:, JointDenis.revmap]
            pose /= 1000.
            # The output is scaled to 2m by Denis.
            # We change this to 1.8 * a scale in order to correct for
            # the skeletons being a bit too high still.
            pose *= 1.8 / 2.
            pose[2, :] *= -1.
            pose = pose[[0, 2, 1], :]
            pose[:, Joint.PELV] = (pose[:, Joint.RHIP] + pose[:, Joint.LHIP]) \
                                  / 2.
            if skel2d is not None:
                skel3d.set_pose(frame_id=frame_id2, pose=pose,
                                time=skel2d.get_time(frame_id))
                for j in range(Skeleton.N_JOINTS):
                    conf = skel2d.get_confidence(frame_id=frame_id2, joint=j)
                    skel3d.set_confidence(frame_id=frame_id2, joint=j, confidence=conf)
                    skel3d.set_visible(frame_id=frame_id2, joint=j, visible=conf > 0.5)

        prev_entry = entry
    except ValueError as e:
        print("Problem for %s: %s" % (fname, e))
    # if len(entries):
    #     break
    if not (fname_id % 10) and len(entries):
        out_name = \
            os.path.join(out_dir, "skeletons_%03d.json" % fname_id)
        with open(out_name, 'w') as f_out:
            json.dump(entries, f_out, default=default_encode, indent=2)
            print("Wrote to %s" % out_name)


out_name = \
    os.path.join(out_dir, "skeletons_%03d.json" % fname_id)
                 # "%s.json" % os.path.splitext(os.path.split(fname)[1])[0])
with open(out_name, 'w') as f_out:
    json.dump(entries, f_out, default=default_encode, indent=2)
    print("Wrote to %s" % out_name)

if skel2d is not None:
    Scenelet(skeleton=skel3d).save(os.path.join(args.dest_dir, 'skel_GT_3d.json'))



