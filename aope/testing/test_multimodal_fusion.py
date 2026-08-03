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

from modules.features.cross_attention_transformer import LearnablePositionalEncoding
from modules.features.cross_attention_transformer import (
    MultiModalCrossAtentionTransformer,
)

from modules.pose_estimation.modules import *
from modules.reconstruction.reconstruction import (
    pad_non_part_points,
    reconstruct_pose_params,
)

from modules.loss.losses import pipeline_loss


def test_1():
    dataset_config = DatasetConfig()
    dataset_config.path = "/home/marek/Desktop/hector_ai/synthetic_data_generation/articulated_objects/output/valve"
    data = AOPEDataset(dataset_conf=dataset_config)

    preprocessing_config = PreprocessingConfig()
    preprocessing_config.preprocess_segmentations = False
    preprocessing_config.save_back_to_disk = False
    preprocessing_config.create_debug_imgs = True
    preprocessing_config.debug_path = (
        "/home/marek/aope_for_deformed_objects/aope/testing/debug"
    )

    data_preprocessor = AOPEPreprocesser(data, preprocessing_config)
    data_preprocessor.run()

    # PCL ##############################################
    pcl_tensor = torch.tensor(
        data_preprocessor.pcls_cam_coords, dtype=torch.float32
    ).to("cuda:0")
    point_pixel_coords = torch.tensor(
        data_preprocessor.pcls_pixel_coords, dtype=torch.float32
    ).to("cuda:0")

    pcl_feat_config = PCLFeatureConfig()
    pcl_feat_config.backbone_config = RANDLAConfig()

    backbone = RANDLABackbone(cfg=pcl_feat_config)

    pcl_feature = backbone.extract_features(pcl_tensor)

    ##IMG ###################################################

    imgs = []
    for idx in range(len(data)):
        img = torch.Tensor(cv2.imread(data.load_img(idx))).unsqueeze(0)
        imgs.append(img)

    imgs = torch.cat(imgs, dim=0)

    v_feat_conf = VisualFeatureConfig()
    v_feat_conf.backbone_config = RadioConfig()

    # preprocessor = FeaturePreprocessor(v_feat_conf)
    # img_features = preprocessor.model.get_local_features(imgs).to("cuda:0")

    img_features = torch.rand((len(imgs), 32 * 32, 1280)).to("cuda:0")

    print("Img device: {}".format(img_features.device))
    print("Pcl device: {}".format(pcl_feature.device))

    torch.autograd.set_detect_anomaly(True, check_nan=False)
    ## TRANSFORMER #####################################

    transformer = MultiModalCrossAtentionTransformer(
        img_features[0].shape[-1],
        img_H=dataset_config.imgsz,
        img_W=dataset_config.imgsz,
        point_emb_dim=pcl_feat_config.backbone_config.feature_dim,
        patch_size=v_feat_conf.backbone_config.patch_size,
        num_layers=3,
    )

    print("Running transformer")

    fused_feats = transformer(
        img_features,
        pcl_feature,
        point_pixel_coords.round().to(torch.int32),
        pcl_tensor.clone(),
    )

    part_assingments = (
        torch.tensor(data_preprocessor.part_assingments, dtype=torch.int64).to("cuda:0")
        - 1
    )

    K = dataset_config.num_parts

    fused_feats = torch.cat((fused_feats, pcl_tensor), dim=-1)

    print("Append xyz to features")
    print(fused_feats.shape)

    ########### Shape Reconstruction #############

    print("Apply shape reconstruction")

    # shape_reconstruction_mlps = [ ShapeReconstruction(fused_feats.shape[1],  fused_feats.shape[-1]).to("cuda:0") for i in range(K)]

    shape_reconstuction_mlp = ShapeReconstruction(
        fused_feats.shape[1], fused_feats.shape[-1], K
    ).to("cuda:0")

    print("Fused feature shape: {}".format(fused_feats.shape))

    deformations_pred = shape_reconstuction_mlp(fused_feats, part_assingments)

    base_deformation = deformations_pred[0]
    part_deformation = deformations_pred[1:]

    joint_pred = JointParameterPediction(
        fused_feats.shape[-1], 16, 6, 3, fused_feats.device
    )

    joint_params = joint_pred(fused_feats, pcl_tensor)

    base_deformation_padded, base_padded = pad_non_part_points(
        base_deformation, pcl_tensor, part_assingments, 0, strategy="Random-Sample"
    )

    part_deformation_padded, part_padded = pad_non_part_points(
        part_deformation[0], pcl_tensor, part_assingments, 1, strategy="Random-Sample"
    )

    part_deformation_padded = part_deformation_padded.unsqueeze(1)
    part_padded = part_padded.unsqueeze(1)

    print("Part padded:")
    print(part_padded.shape)



    pivot_points = joint_params[:, :3].unsqueeze(1).unsqueeze(-1)
    rotation_axis = joint_params[:, 3:].unsqueeze(1)

    print("pivot point")
    print(pivot_points.shape)
    print("rot axis")
    print(rotation_axis.shape)

    (
        predicted_pcl,
        base_rot_residuum,
        canonical_base_center,
        part_rot_residuum,
        part_non_axis_rot,
    ) = reconstruct_pose_params(
        base_deformation_padded,
        base_padded,
        part_deformation_padded,
        part_padded,
        pivot_points,
        rotation_axis,
        part_assingments,
        0,
    )

    pcl_log = o3d.geometry.PointCloud()
    print(predicted_pcl[0].cpu().detach().shape)
    pcl_log.points = o3d.utility.Vector3dVector(predicted_pcl[0].cpu().detach())
    o3d.io.write_point_cloud("test_data/pred_epoch_0.ply", pcl_log)


    pcl_log.points = o3d.utility.Vector3dVector(pcl_tensor[0].cpu().detach())
    o3d.io.write_point_cloud("test_data/target.ply", pcl_log)


    loss = pipeline_loss(
            predicted_pcl,
            base_rot_residuum + part_rot_residuum,
            part_non_axis_rot,
            pcl_tensor,
            canonical_base_center,
        )
    

    loss.backward()


if __name__ == "__main__":
    test_1()
