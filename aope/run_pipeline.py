from config.dataset import DatasetConfig
from config.pipeline import PreprocessingConfig
from modules.segmentation.segmentation import segment
from data_processing.data_loading import AOPEDataset
from data_processing.data_preprocessing import AOPEPreprocesser

if __name__ == "__main__":
    dataset_conf = DatasetConfig()
    dataset_conf.path = "/home/marek/Desktop/hector_ai/synthetic_data_generation/articulated_objects/output/valve"
    dataset = AOPEDataset(dataset_conf)

    print("preprocess")
    preprocessing_conf = PreprocessingConfig()
    preprocessing_conf.debug_path = "/home/marek/Desktop/aope_for_deformed_objects/debug_output"
    preprocessing_conf.segmentation_config.seg_model_config.model_path = "/home/marek/Desktop/hector_ai/model_training/yolo_training/runs/valve_seg9/weights/best.pt"
    preprocessing_conf.save_dir = "/home/marek/Desktop/aope_for_deformed_objects/debug_output"

    preprocesser = AOPEPreprocesser(dataset, preprocessing_conf)
    preprocesser.run()

    print("load preprocessed")
    dataset_conf = DatasetConfig()
    dataset_conf.path =  "/home/marek/Desktop/aope_for_deformed_objects/debug_output/valve_preprocessed"
    dataset = AOPEDataset(dataset_conf, preprocessing_conf)


    