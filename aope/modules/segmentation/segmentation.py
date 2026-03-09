from config.segmentation import SegmentationConfig
from config.segmentation import YOLOSegConfig
from config.segmentation import SegModelTypes


def yolo_seg(images, model_config: YOLOSegConfig):
    import ultralytics

    segmentation_model = ultralytics.YOLO(model_config.model_path)
    return [
        segmentation_model(img, conf=model_config.threshold, imgsz=model_config.imgsz)
        for img in images
    ]


def segment(images, seg_conf: SegmentationConfig):
    if seg_conf.seg_model_type == SegModelTypes.YOLO:
        return yolo_seg(
            images,
            model_config=seg_conf.seg_model_config,
        )
