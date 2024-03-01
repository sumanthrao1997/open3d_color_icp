# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion


class TUMDataset:
    def __init__(self, data_dir: Path, *_, **__):
        self.data_dir = Path(data_dir)

        # Load depth frames
        self.depth_frames = np.loadtxt(fname=self.data_dir / "depth.txt", dtype=str)
        self.rgb_frames = np.loadtxt(fname=self.data_dir / "rgb.txt", dtype=str)
        # matches
        self.matches = self.get_matches(self.depth_frames[:, 0], self.rgb_frames[:, 0])
        # depth scale factor
        self.depth_scale = 5000.0
        # Load GT poses
        gt_list = np.loadtxt(fname=self.data_dir / "groundtruth.txt", dtype=str)
        self.gt_poses = self.load_poses(gt_list)

    def __len__(self):
        #  return len(self.matches)
        return len(self.matches)  # temp remove and uncomment above

    def get_matches(self, src_timstamps, target_timstamps):
        indices = np.abs(
            (
                np.subtract.outer(
                    src_timstamps.astype(np.float64),
                    target_timstamps.astype(np.float64),
                )
            )
        )
        src_matches = np.arange(len(src_timstamps))
        target_matches = np.argmin(indices, axis=1)
        _, unique_indxs = np.unique(target_matches, return_index=True)
        matches = np.vstack((src_matches[unique_indxs], target_matches[unique_indxs])).T
        return matches

    def load_poses(self, gt_list):
        gt_indices = self.get_matches(
            gt_list[:, 0].astype(np.float64),
            self.depth_frames[:, 0][self.matches[:, 0]].astype(np.float64),
        )
        poses = gt_list[gt_indices[:, 0]]
        return poses

    def get_frames_timestamps(self):
        return self.depth_frames[:, 0]

    def get_intrinsics(self) -> np.ndarray:
        #  intrinsics = np.array(([fx, 0, cx], [0, fy, cy], [0, 0, 1]))  # focal length x
        cx = 319.5
        cy = 239.5
        fx = 525.0
        fy = 525.0
        return np.array(([fx, 0, cx], [0, fy, cy], [0, 0, 1]))

    def __getitem__(self, idx):
        depth_idx, rgb_idx = self.matches[idx]
        rgb = cv2.imread(str(self.data_dir / self.rgb_frames[rgb_idx][-1]), cv2.IMREAD_COLOR)
        #  print(str(self.data_dir / self.rgb_frames[rgb_idx][-1]))
        # read images and process depth
        depth_unprocessed = cv2.imread(
            str(self.data_dir / self.depth_frames[depth_idx][-1]), cv2.IMREAD_UNCHANGED
        )
        depth = (depth_unprocessed / self.depth_scale).astype(np.float32)
        return rgb, depth
