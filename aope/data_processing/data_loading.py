from typing import List

import json
import os
import numpy as np
import open3d as o3d

from utility.transforms import T
from config.pipeline import DatasetConfig
from config.pipeline import PreprocessingConfig

from torch.utils.data import Dataset


class AOPEDataset(Dataset):

    def __init__(
        self,
        dataset_conf: DatasetConfig,
        preprocessing_conf: PreprocessingConfig = None,
    ):
        self.config = dataset_conf
        self.label_config = dataset_conf.label_config

        self.pcl_dir = os.path.join(self.config.path, self.config.pcl_dir_name)
        self.pcl_files = [
            os.path.join(self.pcl_dir, file)
            for file in sorted(os.listdir(self.pcl_dir))
            if file.endswith(".ply")
        ]

        self.img_dir = os.path.join(self.config.path, self.config.img_dir_name)
        self.img_files = [
            os.path.join(self.img_dir, file)
            for file in sorted(os.listdir(self.img_dir))
            if file.split(".")[-1] in ["png", "jpg"]
        ]
        self.return_seg = False

        self.labels = {}
        if preprocessing_conf:
            self.load_preprocessed_labels(preprocessing_conf)

            if preprocessing_conf.preprocess_segmentations:
                self.seg_dir = os.path.join(self.config.path, "segmentations")
                self.seg_files = [
                    os.path.join(self.seg_dir, file)
                    for file in sorted(os.listdir(self.seg_dir))
                    if file.endswith(".json")
                ]
                self.return_seg = True
        else:
            self.load_labels()

    def load_pcl(self, i):
        return o3d.io.read_point_cloud(self.pcl_files[i])

    def load_img(self, i) -> str:
        return self.img_files[i]

    def load_segmentation(self, i):
        with open(self.seg_files[i], "r") as seg_file:
            seg = json.load(seg_file)

        return seg

    def load_labels(self) -> None:

        with open(
            os.path.join(self.config.path, self.config.label_file_name), "r"
        ) as label_file:
            labels = json.load(label_file)

        self.labels[self.label_config.cam_intrinsics_label] = [
            np.array(intrinsics)
            for intrinsics in labels[self.label_config.cam_intrinsics_label]
        ]
        self.labels[self.label_config.cam_pose_label] = [
            T.from_matrix(mat=pose) for pose in labels[self.label_config.cam_pose_label]
        ]

        obj_labels = labels[self.label_config.obj_data_label]

        for label in ["obj_pose", "obj_pose_mat", "joints", "joint_states"]:
            self.labels[label] = []

        for obj_label in obj_labels:
            rot = obj_label[self.label_config.obj_pose_label]["rotation"]
            transl = obj_label[self.label_config.obj_pose_label]["translation"]

            self.labels["obj_pose_mat"].append(T.from_euler_xyz(rot, transl))
            self.labels["obj_pose"].append(np.array(rot + transl))

            self.labels["joints"].append(
                list(obj_label[self.label_config.obj_joint_state_label].keys())
            )
            self.labels["joint_states"].append(
                [
                    state
                    for state in obj_label[
                        self.label_config.obj_joint_state_label
                    ].values()
                ]
            )

    def load_preprocessed_labels(self, preprocessing_config: PreprocessingConfig):
        with open(
            os.path.join(self.config.path, self.config.label_file_name), "r"
        ) as label_file:
            labels = json.load(label_file)

            self.labels[self.label_config.cam_intrinsics_label] = [
                np.array(intrinsics)
                for intrinsics in labels[self.label_config.cam_intrinsics_label]
            ]
            self.labels[self.label_config.cam_pose_label] = [
                T.from_matrix(mat=pose)
                for pose in labels[self.label_config.cam_pose_label]
            ]

            self.labels["obj_pose"] = np.array(labels["obj_pose"])
            self.labels["obj_pose_mat"] = [
                T.from_matrix(mat=mat) for mat in labels["obj_pose_mat"]
            ]
            self.labels["joints"] = labels["joints"]
            self.labels["joint_states"] = labels["joint_states"]

            if preprocessing_config.preprocess_segmentations:
                self.labels["part_assingments"] = labels["part_assingments"]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        item = {
            "img": self.load_img(idx),
            "pcl": self.load_pcl(idx),
            **{label: self.labels[label][idx] for label in self.labels},
        }
        
        if self.return_seg:
            item.update(self.load_segmentation(idx))

        return item
