#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""
import tensorflow as tf

import utils.config as config
import utils.process as ut
from utils import cpm
from utils.draw import *
from utils.prob_model import Prob3dPose
import os
import json

import skimage # by Aron for show_heatmaps

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest="image", help="Image to work with")
parser.add_argument('-d', dest="image_dir", help="Folder of images to work with")
parser.add_argument('--no-vis', action="store_true", help="No visualization")
args = parser.parse_args()


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


def load_image(fname):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    scale = config.INPUT_SIZE/(image.shape[0] * 1.0)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    print("image.size: %s" % repr(image.shape))
    b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)
    return b_image, image

inputs = []
if args.image_dir is not None:
    for f in os.listdir(args.image_dir):
        if f.endswith('jpg') or f.endswith('png'):
            inputs.append(os.path.join(args.image_dir, f))
else:
    fname = args.image or 'images/test_image.png'
    inputs.append(fname)

b_image, image = load_image(inputs[0])
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

entries = {"image_shape": shape}
out_dir = None
for fname_id, fname in enumerate(sorted(inputs)):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(fname), os.pardir, "denis")
        print("will write to %s" % out_dir)
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
    b_image, image = load_image(fname)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'saved_sessions/person_MPI/init')
        hmap_person = sess.run(heatmap_person_large, {image_in: b_image})

    hmap_person = np.squeeze(hmap_person)
    centers = ut.detect_objects_heatmap(hmap_person)
    print("hmap_person.shape: %s" % repr(hmap_person.shape))
    b_pose_image, b_pose_cmap = \
        ut.prepare_input_posenet(b_image[0], centers,
                                 [config.INPUT_SIZE, shape[1]],
                                 [config.INPUT_SIZE, config.INPUT_SIZE])
    print("b_pose_image.shape: %s" % repr(b_pose_image.shape))
    print("b_bose_cmap.shape: %s" % repr(b_pose_cmap.shape))


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
    parts, visible = \
        ut.detect_parts_heatmaps(_hmap_pose, centers,
                                 [config.INPUT_SIZE, config.INPUT_SIZE])
    # show_heatmaps(_hmap_pose, centers,
    #               [config.INPUT_SIZE, config.INPUT_SIZE])

    try:
        # Estimate 3D poses
        poseLifting = Prob3dPose()
        pose2D, weights = Prob3dPose.transform_joints(parts, visible)
        pose3D = poseLifting.compute_3d(pose2D, weights)

        if not args.no_vis:
            # Show 2D poses
            plt.figure()
            draw_limbs(image, parts, visible)
            plt.imshow(image)
            plt.axis('off')

            # Show 3D poses
            for single_3D in pose3D:
                plot_pose(poseLifting.centre_all(single_3D))

            plt.show()

        entry = {'pose_2d': parts.tolist(),
                 'visible': visible.astype(int).tolist(),
                 'pose_3d': pose3D.tolist(),
                 'centered_3d': [poseLifting.centre_all(single_3D).tolist()
                                 for single_3D in pose3D]}
        entries[os.path.splitext(os.path.split(fname)[1])[0]] = entry
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
    os.path.join(out_dir, "skeletons.json")
                 # "%s.json" % os.path.splitext(os.path.split(fname)[1])[0])
with open(out_name, 'w') as f_out:
    json.dump(entries, f_out, default=default_encode, indent=2)
    print("Wrote to %s" % out_name)





