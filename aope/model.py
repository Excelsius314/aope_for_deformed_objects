import torch

from config.features import PCLFeatureConfig, VisualFeatureConfig

from modules.features.cross_attention_transformer import LearnablePositionalEncoding
from modules.features.cross_attention_transformer import (
    MultiModalCrossAtentionTransformer,
)

from modules.pose_estimation.modules import *
from modules.reconstruction.reconstruction import (
    pad_non_part_points,
    reconstruct_pose_params,
)
from modules.features.pcl_features import RANDLABackbone
from modules.features.randla_net.randla_model import SmallRandla

from modules.loss.losses import canonical_consistency_loss

from utility.transforms import homogenous_transform

def feature_collapse_stats(feats, mask=None, name=""):
    """
    feats: (B, N, C) tensor — per-observation features before or after fusion
    mask:  optional (B, N) bool tensor, e.g. base_mask or part_mask,
           to restrict the stat to relevant points
    name:  label for logging

    Returns a dict of scalars describing how much variation exists
    across the batch dimension (B) — i.e. how distinguishable
    observations still are from one another.
    """
    if mask is not None:
        # Keep batch dim, select only masked points.
        # Assumes mask selects the same number of points per sample;
        # if not, you'll need to handle ragged selection separately.
        feats = feats[mask].reshape(feats.shape[0], -1, feats.shape[-1])

    # Variance across the batch dimension, for each point and each channel.
    # Shape: (N, C)
    var_per_point_channel = feats.var(dim=0, unbiased=False)

    # A few useful reductions:
    mean_var = var_per_point_channel.mean().item()      # overall collapse scalar
    max_var = var_per_point_channel.max().item()         # best surviving channel/point
    min_var = var_per_point_channel.min().item()         # most collapsed channel/point

    # Also useful: average pairwise distance between samples in the batch,
    # which is a more intuitive "are these observations different at all" check.
    B = feats.shape[0]
    flat = feats.reshape(B, -1)  # (B, N*C)
    if B > 1:
        diffs = flat.unsqueeze(0) - flat.unsqueeze(1)  # (B, B, N*C)
        pairwise_dist = diffs.norm(dim=-1)             # (B, B)
        avg_pairwise_dist = pairwise_dist[~torch.eye(B, dtype=torch.bool, device=feats.device)].mean().item()
    else:
        avg_pairwise_dist = float("nan")

    stats = {
        f"{name}_mean_var_across_batch": mean_var,
        f"{name}_max_var_across_batch": max_var,
        f"{name}_min_var_across_batch": min_var,
        f"{name}_avg_pairwise_dist": avg_pairwise_dist,
    }
    return stats

import torch
import torch.nn as nn

class PartEncoding(nn.Module):
    def __init__(self, n_parts, embed_dim):
        """
        n_parts:   number of distinct parts (K), e.g. 2 for base/handle
        embed_dim: dimensionality of the part embedding.
                    Common choices: same as feature_dim (for additive fusion)
                    or smaller (for concat fusion)
        """
        super().__init__()
        self.embedding = nn.Embedding(n_parts, embed_dim)

    def forward(self, part_assignment):
        """
        part_assignment: (B, N) integer tensor, values in [0, n_parts)
        returns: (B, N, embed_dim)
        """
        return self.embedding(part_assignment)


class AOPEModel(nn.Module):

    def __init__(
        self,
        dataset_conf,
        v_feat_conf: VisualFeatureConfig,
        pcl_feat_config: PCLFeatureConfig,
        device,
    ):
        super(AOPEModel, self).__init__()

        self.K = dataset_conf.num_parts
        self.device = device

        self.pcl_back_bone = RANDLABackbone(pcl_feat_config)

        self.transformer = MultiModalCrossAtentionTransformer(
            v_feat_conf.backbone_config.embedding_dim,
            img_H=dataset_conf.imgsz,
            img_W=dataset_conf.imgsz,
            point_emb_dim=pcl_feat_config.backbone_config.feature_dim,
            patch_size=v_feat_conf.backbone_config.patch_size,
            num_layers=2,
            device=device,
            out_dim=pcl_feat_config.backbone_config.feature_dim
        )

        self.part_encoder = PartEncoding(2, 8)#pcl_feat_config.backbone_config.feature_dim)

        #self.feature_bottleneck = SmallRandla(
        #    d_in=64,
        #    num_neighbors=pcl_feat_config.backbone_config.num_neighbors,
        #    decimation=pcl_feat_config.backbone_config.decimation,
        #    feature_dim=pcl_feat_config.backbone_config.feature_dim,
        #    device=torch.device(self.device)
        #)

        self.shape_reconstuction_mlp = ShapeReconstruction(
            dataset_conf.pcl_size,
            #pcl_feat_config.backbone_config.feature_dim + 3,
            pcl_feat_config.backbone_config.feature_dim,
            self.K,
        )

        self.joint_params_head = JointParameterPediction(
            # pcl_feat_config.backbone_config.feature_dim + 3, 16, 6, 3, device
            # pcl_feat_config.backbone_config.feature_dim , 16, 6, 3, device
            #pcl_feat_config.backbone_config.feature_dim + 3,
            pcl_feat_config.backbone_config.feature_dim,
            pcl_feat_config.backbone_config.num_neighbors,
            pcl_feat_config.backbone_config.decimation,
            3,
            device,
        )

    def forward(self, input):

        ##### Setup Data #####
        pcl_tensor = input["pcls_cam_coords"]
        part_assingments = input["part_assingments"] #-1
        point_pixel_coords = input["pcls_pixel_coords"]
        img_features = input["img_features"]

        #pcl_tensor = homogenous_transform(input["cam_poses"], pcl_tensor)
        pcl_tensor = input["pcl"]

        ################ Extract features ##############

        pcl_feature = self.pcl_back_bone.extract_features(pcl_tensor, self.part_encoder(part_assingments))

        #pcl_feature = pcl_feature + self.part_encoder(part_assingments)

        fused_feats = self.transformer(
            img_features,
            pcl_feature,
            point_pixel_coords.round().to(torch.int32),
            pcl_tensor.clone(),
        )

        base_mask = part_assingments == 0
        part_mask = part_assingments == 1

        test_feats = fused_feats[0]

        #fused_feats = fused_feats.mean(dim=0).unsqueeze(0).expand(pcl_feature.shape)

        intermediate_feats_loss = canonical_consistency_loss(fused_feats, None)

        if fused_feats.shape[0] > 1:
            base_feats = fused_feats[base_mask, :].reshape((base_mask.shape[0], base_mask.shape[1] // 2, 128))
            fused_feats[base_mask, :] = base_feats.mean(dim=0).unsqueeze(0).expand(pcl_feature.shape[0], base_feats.shape[1], pcl_feature.shape[2]).reshape((-1,pcl_feature.shape[2]) )

            part_feats = fused_feats[part_mask, :].reshape((base_mask.shape[0], base_mask.shape[1] // 2, 128))
            fused_feats[part_mask, :] = part_feats.mean(dim=0).unsqueeze(0).expand(pcl_feature.shape[0], base_feats.shape[1], pcl_feature.shape[2]).reshape((-1,pcl_feature.shape[2]) )

        #fused_feats_coords = torch.cat((fused_feats, pcl_tensor), dim=-1)



        ########### Shape Reconstruction #############
        deformations_pred = self.shape_reconstuction_mlp(fused_feats, part_assingments)

        base_deformation = deformations_pred[0]
        part_deformation = deformations_pred[1:]

        print("base deform shape: {}".format(base_deformation.shape))

        test_canonical = self.shape_reconstuction_mlp(test_feats.unsqueeze(0).detach(), part_assingments[0].unsqueeze(0).detach())[0].cpu()

        joint_params = self.joint_params_head(fused_feats, pcl_tensor)

        return (
            pcl_tensor,
            part_assingments,
            base_deformation,
            part_deformation,
            joint_params,
            intermediate_feats_loss,
            test_canonical 
        )

    def reconstruct(
        self,
        pcl_tensor,
        part_assingments,
        base_deformation,
        part_deformation,
        joint_params,
    ):

        ######### Padding #########
        base_deformation_padded, base_padded = pad_non_part_points(
            base_deformation, pcl_tensor, part_assingments, 0, strategy="Random-Sample"
        )

        part_deformation_padded, part_padded = pad_non_part_points(
            part_deformation[0],
            pcl_tensor,
            part_assingments,
            1,
            strategy="Random-Sample",
        )

        part_deformation_padded = part_deformation_padded.unsqueeze(1)
        part_padded = part_padded.unsqueeze(1)

        pivot_points = joint_params[:, :3].unsqueeze(1).unsqueeze(-1)
        rotation_axis = joint_params[:, 3:].unsqueeze(1)

        return reconstruct_pose_params(
            base_deformation_padded,
            base_padded,
            part_deformation_padded,
            part_padded,
            pivot_points,
            rotation_axis,
            part_assingments,
            0,
        )
