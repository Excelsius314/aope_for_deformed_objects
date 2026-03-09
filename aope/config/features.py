from omegaconf import DictConfig, OmegaConf
from typing import Any
from dataclasses import dataclass
from enum import Enum

class VisBackBoneModels(Enum):
    RADIO=0
    DINO=1
    DINO_FEATUP=2

@dataclass
class VisBackboneConfig:
    model_version = None
    patch_size: int = None
    backbone_type = None

@dataclass
class RadioConfig(VisBackboneConfig):
    model_version: str = "nvidia/RADIO"
    patch_size: int = 14
    backbone_type: VisBackBoneModels = VisBackBoneModels.RADIO

@dataclass
class DINOConfig(VisBackboneConfig):
    model_version: str = "dino_vitb16"
    patch_size: int = 16
    backbone_type: VisBackBoneModels = VisBackBoneModels.DINO
    batch_size: int = 1

@dataclass
class DINOFeatupConfig(VisBackboneConfig):
    model_version: str = "dino_vitb16"
    patch_size: int = 14
    backbone_type: VisBackBoneModels = VisBackBoneModels.DINO_FEATUP
    max_input_size: int = 518

class PCLBackBoneModels(Enum):
    RANDLA=0

@dataclass
class PCLBackboneConfig:
    d_in: int = 3

@dataclass
class RANDLAConfig(PCLBackboneConfig):
    decimation : int = 4
    num_neighbors : int = 16

@dataclass
class PCLFeatureConfig:
    backbone_config : PCLBackboneConfig = PCLBackboneConfig


class VisualFeatureInputType(Enum):
    PART_SPECIFIC=1
    WHOLE_IMAGE=2

class VisualFeatureOutputType(Enum):
    LOCAL=1
    GLOBAL=2
    BOTH=3



@dataclass
class VisualFeatureConfig:
    backbone_config: VisBackboneConfig = VisBackboneConfig
    feature_input_type: VisualFeatureInputType = VisualFeatureInputType.WHOLE_IMAGE
    feature_output_type: VisualFeatureOutputType = VisualFeatureOutputType.LOCAL

@dataclass
class VisualFeatureBranchConfig:
    use_pretrained: bool = True
    use_preprocessed_segmentation = True

    output_dim: int = 512
