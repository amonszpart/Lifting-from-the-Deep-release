import os
import sys
import json
import cv2
import shutil
import numpy as np

np.set_printoptions(suppress=True)


class JointOpenPose(object):
    # Ours (stealth.logic.joints.Joints):
    #  RANK = 0
    #  RKNE = 1
    #  RHIP = 2
    #  LHIP = 3
    #  LKNE = 4
    #  LANK = 5
    #  PELV = 6
    #  THRX = 7
    #  NECK = 8
    #  HEAD = 9
    #  RWRI = 10
    #  RELB = 11
    #  RSHO = 12
    #  LSHO = 13
    #  LELB = 14
    #  LWRI = 15

    _op_2_ours_2d = {
        # 0: ?, # NOSE/NECK
        1: 7, # THRX
        2: 12, # RSHO
        3: 11, # RELB
        4: 10, # RWRI
        5: 13, # LSHO
        6: 14, # LELB
        7: 15, # LWRI
        8: 2, # RHIP
        9: 1, # RKNE
        10: 0, # RANK
        11: 3, # LHIP
        12: 4, # LKNE
        13: 5 # LANK
        # 14: ?, # REYE
        # 15: ?, # LEYE
        # 16: ?, # REAR
        # 17: ?, # LEAR
    }
    """Openpose to ours."""

    _op_2_lfd_2d = {
        0: 0, # NOSE/NECK visually !! 9 is Joints.HEAD
        1: 1, # THRX
        2: 2, # RSHO
        3: 3, # RELB
        4: 4, # RWRI
        5: 5, # LSHO
        6: 6, # LELB
        7: 7, # LWRI
        8: 8, # RHIP
        9: 9, # RKNE
        10: 10, # RANK
        11: 11, # LHIP
        12: 12, # LKNE
        13: 13  # LANK
        # 14: ?, # REYE
        # 15: ?, # LEYE
        # 16: ?, # REAR
        # 17: ?, # LEAR
    }

    _revmap_lfd_2d = [k for k, v in sorted(list(_op_2_lfd_2d.items()),
                                           key=lambda e: e[1])]

    @classmethod
    def op_to_lfd(cls, keypoints, out=None):
        op_2_lfd_2d = cls._op_2_lfd_2d
        do_ret = False
        if out is None:
            out = np.zeros(shape=(14, 2))
            do_ret = True
        else:
            assert out.shape == (14, 2), "no: %s" % repr(out.shape)

        out[0, ...] = keypoints[0, ...]
        for jid in range(1, 14):
            jid2 = op_2_lfd_2d[jid]
            out[jid, ...] = keypoints[jid2, ...]

        if do_ret:
            return out

    @classmethod
    def parse_keypoints(cls, people, conf_thresh=0.1):
        lfd_2d = np.zeros(shape=(len(people), 14, 2))
        visible = np.zeros(shape=(len(people), 14), dtype='b1')
        visible_f = np.zeros(shape=(len(people), 14))
        revmap = cls._revmap_lfd_2d
        print(revmap)
        for pid, person in enumerate(people):
            kp = np.array(person['pose_keypoints_2d']).reshape((-1, 3))
            JointOpenPose.op_to_lfd(keypoints=kp[:, :2], out=lfd_2d[pid, ...])
            visible_f[pid, :] = kp[revmap, 2]
            visible[pid, :] = kp[revmap, 2] > conf_thresh
        print(lfd_2d)
        print(lfd_2d.shape)
        print(visible)

        return lfd_2d, visible, visible_f

def main(argv):
    scene_root = "/media/data/amonszpa/stealth/shared/video_recordings/angrymen00"

    dest = os.path.join(scene_root, "debug3")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)

    im_id = 1
    p = "%s/openpose_keypoints/color_%05d_keypoints.json" % (scene_root, im_id)
    data = json.load(open(p, 'r'))
    print("data: %s" % data)

    p2 = "%s/origjpg/color_%05d.jpg" % (scene_root, im_id)
    im = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)

    lfd_2d = np.zeros(shape=(len(data['people']), 14, 2))
    for pid, person in enumerate(data['people']):
        kp = np.array(person['pose_keypoints_2d']).reshape((-1, 3))
        person_lfd = JointOpenPose.op_to_lfd(kp[:, :2])
        lfd_2d[pid, ...] = person_lfd
        for i, row in enumerate(kp):
            if row[2] < 0.1:
                continue
            pos = (int(round(row[0])), int(round(row[1])))
            cv2.circle(im, center=pos, radius=3, color=(128, 128, 0))
            cv2.putText(im, "%d" % pid, org=pos, fontFace=1, fontScale=1,
                        color=(128, 0, 128), thickness=2)
        print(kp)

    lfd_2d = JointOpenPose.parse_keypoints(data['people'])

    p_dest = os.path.join(dest, "op_%05d.jpg" % im_id)
    cv2.imwrite(p_dest, im)
    print("wrote to %s" % p_dest)

if __name__ == '__main__':
    main(sys.argv[1:])