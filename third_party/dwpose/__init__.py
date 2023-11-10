# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return {
                "rgb": draw_pose(pose, H, W),
                "dict": pose
            }


class DWPoseBatchInfer(object):
    def __init__(self):
        self.processor = DWposeDetector()
        self.latest_keypoint = {
            "version": "dwpose",
            "people": [
                {
                    "person_id": [-1],
                    "person_keypoints_2d": [],
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }
            ]
        }

    def forward_rgb_as_rgb(self, x_arr: np.ndarray) -> np.ndarray:
        assert len(x_arr.shape) == 3, "Only HWC is supported"
        detected_res = self.processor(x_arr)
        detected_map = detected_res["rgb"]
        detected_dict = detected_res["dict"]
        self.update_latest_keypoint_dict(detected_dict)
        return detected_map

    def update_latest_keypoint_dict(self, pose: dict):
        bodies = pose["bodies"]  # dict["candidate"]
        hands = pose["hands"]  # np.ndarray
        faces = pose["faces"]  # np.ndarray

        self.latest_keypoint["people"][0]["person_keypoints_2d"] = pose["bodies"]
        self.latest_keypoint["people"][0]["face_keypoints_2d"] = pose["faces"]
        self.latest_keypoint["people"][0]["hand_left_keypoints_2d"] = pose["hands"]
        self.latest_keypoint["people"][0]["hand_right_keypoints_2d"] = pose["hands"]

        # no need to save POSE.yaml yet

    def get_latest_keypoint_dict(self):
        return self.latest_keypoint
