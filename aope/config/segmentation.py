
from dataclasses import dataclass
from enum import Enum

@dataclass
class SegModelConfig:
    model_path: str = "segmentation_model.pth"
    threshold: float = 0.6

@dataclass
class YOLOSegConfig(SegModelConfig):
    imgsz = 512

class SegModelTypes(Enum):
    YOLO = 1

@dataclass
class SegmentationConfig:
    seg_model_type : SegModelTypes = SegModelTypes.YOLO
    seg_model_config = YOLOSegConfig
