from modules.features.visual_features import VisualFeaturePreprocessor
from modules.features.visual_features import HFModel

from config.features import *
import cv2
import os
import torch
from PIL import Image

if __name__ == "__main__":

    branch_conf = VisualFeatureBranchConfig()

    feature_conf = VisualFeatureConfig()

    im_dir = "/home/marek/Desktop/aope_for_deformed_objects/debug_output/valve_preprocessed/images"

    v_feat_conf = VisualFeatureConfig()
    # v_feat_conf.backbone_config.model_version = "nvidia/E-RADIO"

    v_feat_conf.backbone_config.backbone_type = VisBackBoneModels.DINO_FEATUP
    v_feat_conf.backbone_config.model_version = "mhamilton723/FeatUp"

    preprocessor = VisualFeaturePreprocessor(v_feat_conf)

    imgs = [
        cv2.imread(os.path.join(im_dir, img_file))
        for img_file in sorted(os.listdir(im_dir))
    ]

    #local_features = preprocessor.model.get_local_features(imgs)

    #preprocessor.model.pca_vis(local_features, title="dinov2_pca_img")

    #use_norm = True
    #upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).cuda()

    #image_tensors = [ Image.open(os.path.join(im_dir, img_file)).convert("RGB") for img_file in sorted(os.listdir(im_dir))]
    #image_tensor = transform(image_tensors[0]).unsqueeze(0).cuda()

    #print(image_tensor.shape)
    #hr_feats = upsampler(image_tensor).detach()

    #print(hr_feats.shape)

    #hr_feats = torch.flatten(hr_feats, 2, 3)
    #hr_feats = torch.swapaxes(hr_feats, 1, 2)
    #print(hr_feats.shape)

    features = preprocessor.model.get_local_features(imgs)
    
    HFModel.pca_vis(features, title="upsampled_dinov2_pca_img")



    # summary, features = preprocessor.model.get_features(imgs)

    # torch.save(features, '/home/marek/Desktop/aope_for_deformed_objects/debug_output/feat.pt')

    # features = torch.load('/home/marek/Desktop/aope_for_deformed_objects/debug_output/feat.pt')
    # print(features.shape)

    # preprocessor.model.visualize_using_pca(features)
