from dataclasses import dataclass
from enum import Enum


class DatasetFormat(Enum):
    YOLO = "yolo"
    CUSTOM = "custom"


@dataclass
class LabelConfig:
    cam_pose_label = "cam_poses"
    cam_intrinsics_label = "cam_intrinsics"
    obj_data_label = "object_poses"
    obj_pose_label = "pose"
    obj_joint_state_label = "joint_states"
    obj_joint_state_degrees = True  # if false, radians are assumed


@dataclass
class DatasetConfig:
    dataset_format: DatasetFormat = DatasetFormat.YOLO
    path = "data/"
    pcl_dir_name = "point_clouds"
    img_dir_name = "images"

    label_config = LabelConfig
    label_file_name = "scene_data.json"

    imgsz = 512


