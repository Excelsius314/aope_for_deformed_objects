from dataclasses import dataclass
from enum import Enum


class DatasetFormat(Enum):
    YOLO = "yolo"
    CUSTOM = "custom"


@dataclass
class LabelConfig:
    cam_pose_label = "cam_poses"
    cam_intrinsics_label = "cam_intrinsics"
    obj_6D_poses_label = "object_6D_poses"
    obj_geometric_pose_label = "geometric_pose"

    obj_joint_state_label = "joint_states"
    obj_joint_state_degrees = True  # if false, radians are assumed


@dataclass
class DatasetConfig:
    dataset_format: DatasetFormat = DatasetFormat.YOLO
    path = "data/"
    pcl_dir_name = "point_clouds"
    img_dir_name = "images"

    label_config = LabelConfig
    label_file_name = "scene_data.zarr"

    pcl_size = 3500
    imgsz = 512

    num_parts = 2




