from typing import List

import json
import os
import numpy as np
import open3d as o3d

import gzip
import tempfile

from utility.transforms import T
from config.pipeline import DatasetConfig
from config.pipeline import PreprocessingConfig

from torch.utils.data import Dataset

import zarr
import torch


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
            if file.endswith(".ply.gz")
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

        with gzip.open(self.pcl_files[i], "rb") as f:
            ply_bytes = f.read()

        with tempfile.NamedTemporaryFile(suffix=".ply") as tmp:
            tmp.write(ply_bytes)
            tmp.flush()
            pcl = o3d.io.read_point_cloud(tmp.name)

        if self.config.pcl_size > len(pcl.points):
            padding = o3d.geometry.PointCloud()
            padding.points = o3d.utility.Vector3dVector(
                pcl.get_center() * np.ones((self.config.pcl_size - len(pcl.points), 3))
            )
            pcl.append(padding)
        else:
            pcl = pcl.farthest_point_down_sample(self.config.pcl_size)

        return pcl

    def load_img(self, i) -> str:
        return self.img_files[i]

    def load_segmentation(self, i):
        with open(self.seg_files[i], "r") as seg_file:
            seg = json.load(seg_file)

        return seg

    def load_labels(self) -> None:

        # with open(
        #    os.path.join(self.config.path, self.config.label_file_name), "r"
        # ) as label_file:
        #    labels = json.load(label_file)

        z_group = zarr.open_group(
            os.path.join(self.config.path, self.config.label_file_name), mode="r"
        )

        self.labels = z_group
        # self.labels[self.label_config.cam_intrinsics_label] = [
        #    np.array(intrinsics)
        #    for intrinsics in labels[self.label_config.cam_intrinsics_label]
        # ]
        # self.labels[self.label_config.cam_pose_label] = [
        #    T.from_matrix(mat=pose) for pose in labels[self.label_config.cam_pose_label]
        # ]

        return

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

        cam_intrcs_label = self.config.label_config.cam_intrinsics_label
        cam_pose_label = self.config.label_config.cam_pose_label
        obj_6D_poses_label = self.label_config.obj_6D_poses_label
        obj_geometric_pose_label = self.label_config.obj_geometric_pose_label

        # rot = obj_label[self.label_config.obj_pose_label]["rotation"]
        #    transl = obj_label[self.label_config.obj_pose_label]["translation"]

        #    self.labels["obj_pose_mat"].append(T.from_euler_xyz(rot, transl))
        #    self.labels["obj_pose"].append(np.array(rot + transl))

        #    self.labels["joints"].append(
        #        list(obj_label[self.label_config.obj_joint_state_label].keys())
        #    )

        obj_rot = self.labels[obj_6D_poses_label][obj_geometric_pose_label]["rotation"][
            idx
        ]
        obj_xyz = self.labels[obj_6D_poses_label][obj_geometric_pose_label][
            "translation"
        ][idx]

        joint_labels = self.labels[obj_6D_poses_label]["joint_states"]
        joint_names = [joint for joint in joint_labels]

        joint_states = np.array([joint_labels[name][idx] for name in joint_names])

        item = {
            "img": self.load_img(idx),
            "pcl": self.load_pcl(idx),
            cam_intrcs_label: self.labels[cam_intrcs_label][idx],
            cam_pose_label: T.from_matrix(self.labels[cam_pose_label][idx]),
            "obj_pose_mat": T.from_euler_xyz(obj_rot, obj_xyz),
            "obj_pose": np.concat((obj_rot, obj_xyz), axis=-1),
            "joints": joint_names,
            "joint_states": joint_states,
        }

        if self.return_seg:
            item.update(self.load_segmentation(idx))

        return item


class ProcessedAOPEDataset(Dataset):

    def __init__(
        self,
        dataset_conf: DatasetConfig,
        preprocessing_conf: PreprocessingConfig = None,
        device="cuda:0",
    ):
        self.config = dataset_conf
        self.device = device

        self.data = zarr.open_group(os.path.join(self.config.path, ""), mode="r")

        self.data_labels = [
            "pcls_cam_coords",
            "pcls_pixel_coords",
            "part_assingments",
            "img_features",
            "cam_poses",
            "obj_poses",
            "joint_states",
            "data_idx"
        ]

    def __getitem__(self, idx):
        return {
            "pcls_cam_coords": torch.from_numpy(self.data["pcls_cam_coords"][idx]).to(
                dtype=torch.float32
            ),
            "pcls_pixel_coords": torch.from_numpy(
                self.data["pcls_pixel_coords"][idx]
            ).to(dtype=torch.float32),
            "part_assingments": torch.from_numpy(self.data["part_assingments"][idx]).to(
                dtype=torch.int64
            )
            - 1,
            "img_features": torch.from_numpy(self.data["img_features"][idx]).to(
                dtype=torch.float32
            ),
            "cam_poses": torch.from_numpy(self.data["cam_poses"][idx]).to(
                dtype=torch.float32
            ),
            "obj_poses": torch.from_numpy(self.data["obj_poses"][idx]).to(
                dtype=torch.float32
            ),
            "joint_states": torch.from_numpy(self.data["joint_states"][idx]).to(
                dtype=torch.float32
            ),
            "data_idx": idx,
        }

    def collate_fn(self, samples: list[dict]) -> dict:
        return {
            label: (
                torch.stack([s[label] for s in samples], dim=0).to(self.device)
                if label != "data_idx"
                else [s[label] for s in samples]
            )
            for label in self.data_labels
        }

    def __len__(self):
        return len(os.listdir(os.path.join(self.config.path, "point_clouds")))
