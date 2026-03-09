from config.features import *
import os
import torch
import time

import open3d as o3d

from modules.features.pcl_features import RANDLABackbone

if __name__ == "__main__":
    pcl_dir = os.path.join("/home/marek/Desktop/aope_for_deformed_objects/debug_output/valve_preprocessed", "point_clouds")

    feat_config = PCLFeatureConfig()
    feat_config.backbone_config = RANDLAConfig()

    backbone = RANDLABackbone(
        cfg=feat_config
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for pcl_file in sorted(os.listdir(pcl_dir)):
        if not pcl.endswith(".ply"):
            continue

        pcl = o3d.io.read_point_cloud(os.path.join(pcl_dir, pcl_file))
        pcl = torch.tensor(pcl.points).unsqueeze(0)

        feat = backbone.extract_features(pcl.to(device))

        print(feat.shape)