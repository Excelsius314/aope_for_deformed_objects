from dataclasses import dataclass
from enum import Enum
from config.dataset import DatasetConfig
from config.segmentation import SegmentationConfig

@dataclass
class InputConfig:
    data_path: str = "data/"
    shuffle: bool = True

@dataclass
class PreprocessingConfig:
    flip_cam_x_axis = True # Is necesarry to get correct pixel projection if data generation framework uses uncommon cam axis
    preprocess_segmentations = True
    preprocess_visual_features = True
    segmentation_config = SegmentationConfig()

    batch_size = 10

    max_samples = 20000

    create_debug_imgs = False
    debug_path = "debug/"

    save_back_to_disk = True
    save_dir = "data/"

@dataclass
class PipelineConfig:
    dataset_config : DatasetConfig = DatasetConfig
    preprocessing_config : PreprocessingConfig = PreprocessingConfig

   