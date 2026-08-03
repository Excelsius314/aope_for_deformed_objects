import open3d as o3d
from torch import Tensor
import torch
import cv2

import sys
import os

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(ROOT)

from config.dataset import *
from config.pipeline import *
from config.features import *

from modules.features.visual_features import VisualFeaturePreprocessor

from modules.features.pcl_features import RANDLABackbone

from data_processing.data_loading import AOPEDataset
from data_processing.data_preprocessing import AOPEPreprocesser


def preprocess_data():
    dataset_config = DatasetConfig()
    dataset_config.path = "/home/marek/Desktop/hector_ai/synthetic_data_generation/articulated_objects/output/valve_full"
    dataset_config.pcl_size = 3500
    data = AOPEDataset(dataset_conf=dataset_config)
    
    preprocessing_config = PreprocessingConfig()
    preprocessing_config.preprocess_segmentations = False
    preprocessing_config.save_back_to_disk = True
    preprocessing_config.create_debug_imgs = False
    preprocessing_config.debug_path = "/home/marek/Desktop/valve_render_Images"
    preprocessing_config.max_samples = 10000

    v_feat_conf = VisualFeatureConfig()
    v_feat_conf.backbone_config = RadioConfig()

    data_preprocessor = AOPEPreprocesser(data, preprocessing_config, v_feat_conf)
    data_preprocessor.run()


if __name__ == "__main__":
    preprocess_data()
