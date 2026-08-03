import torch
from torch.utils.data import DataLoader

import math

from model import AOPEModel
from data_processing.data_loading import *
from utility.logging import log_validation

from config.features import *
from config.dataset import *
from config.train import *

from tqdm import tqdm

from utility.transforms import homogenous_transform, pivot_transform
from train import random_permute_points_shared

device = "cuda:0"

dataset_conf = DatasetConfig()
dataset_conf.path = (
    "/home/marek/aope_for_deformed_objects/aope/data/valve_full_preprocessed_5"
)
dataset_conf.pcl_size = 512

train_conf = TrainConfig()
train_conf.run_name = "permutation"
train_conf.batch_size = 30

val_epochs = 1

visual_feature_conf = VisualFeatureConfig()
visual_feature_conf.backbone_config = RadioConfig()
pcl_feat_conf = PCLFeatureConfig()
pcl_feat_conf.backbone_config = RANDLAConfig()


model = AOPEModel(dataset_conf, visual_feature_conf, pcl_feat_conf, device).to(device)
model.load_state_dict(
    torch.load(
        "/home/marek/aope_for_deformed_objects/aope/runs/permutation_v4_Jun_21_04_39/weights/weights_epoch_310",
        weights_only=True,
    )
)

model.eval()

dataset = ProcessedAOPEDataset(dataset_conf, device)
loader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
    collate_fn=dataset.collate_fn,
)

base_model = o3d.io.read_point_cloud(
    "/home/marek/aope_for_deformed_objects/aope/data/valve_base.ply"
).farthest_point_down_sample(dataset_conf.pcl_size // 2)
base_model = torch.tensor(base_model.points).to("cuda:0").to(dtype=torch.float32)
part_model = o3d.io.read_point_cloud(
    "/home/marek/aope_for_deformed_objects/aope/data/valve_handle.ply"
).farthest_point_down_sample(dataset_conf.pcl_size // 2)
part_model = torch.tensor(part_model.points).to("cuda:0").to(dtype=torch.float32)
pcl_model = {"base": base_model, "part": part_model}

use_full_obs = True

eval_name = "partial"
eval_log_dir = os.path.join("/home/marek/aope_for_deformed_objects/aope/data/eval", eval_name)

os.makedirs(eval_log_dir, exist_ok=True)


for i, batch in tqdm(
        enumerate(loader),
        total=2,
        desc="Evaluating samples".format(len(loader)),
    ):

        pcl_shape = batch["pcls_cam_coords"].shape

        if i == 3:
            break


        if use_full_obs:
            thetas = (
                (batch["joint_states"] / 360 * 2 * math.pi)
                .to(dtype=torch.float32)
                .unsqueeze(1)
            )
            pivot = (
                torch.tensor([0, 30.5, 0])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(pcl_shape[0], 1, 3)
                .to("cuda:0")
                .to(dtype=torch.float32)
            )
            rot_axis = (
                torch.tensor([0, 1.0, 0.0])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(pcl_shape[0], 1, 3)
                .to("cuda:0")
                .to(dtype=torch.float32)
            )

            part_model = pcl_model["part"]
            part_pcl = pivot_transform(
                pivot,
                rot_axis,
                thetas,
                part_model.expand((pcl_shape[0], pcl_shape[1] // 2, 3)),
            )

            base_pcl = homogenous_transform(
                batch["obj_poses"],
                pcl_model["base"].expand((pcl_shape[0], pcl_shape[1] // 2, 3)),
            )
            part_pcl = homogenous_transform(batch["obj_poses"], part_pcl)
            # base_pcl = homogenous_transform(torch.eye(4).unsqueeze(0).expand((2, 4, 4)).to("cuda:0").to(dtype=torch.float32), pcl_model["base"].expand((pcl_shape[0], pcl_shape[1]//2, 3)))

            part_assingments = torch.zeros((pcl_shape[0], pcl_shape[1]), dtype=torch.int).to("cuda:0")
            part_assingments[:, pcl_shape[1] // 2 :] = 1

            combined_pcl = torch.zeros(pcl_shape).to("cuda:0").to(dtype=torch.float32)
            combined_pcl[:, : pcl_shape[1] // 2, :] = base_pcl
            combined_pcl[:, pcl_shape[1] // 2 :, :] = part_pcl

            pixel_coords = batch["pcls_pixel_coords"]

            combined_pcl, pixel_coords, part_assingments = random_permute_points_shared(
                combined_pcl,  # (B, N, 3)
                pixel_coords,  # (B, N, 3)
                part_assingments,  # (B, N)  -- note: 2D, gather still works since expand handles it
            )

            batch["pcl"] = combined_pcl
            batch["part_assingments"] = part_assingments
            batch["pcls_pixel_coords"] = pixel_coords

            part_mask = batch["part_assingments"] == 1



        # batch["pcl"] = homogenous_transform(batch["obj_poses"], pcl_model.expand((pcl_shape[0], pcl_shape[1], 3)))

        #batch["pcl"] = homogenous_transform(batch["obj_poses"], batch["pcls_cam_coords"]) * 10

        # Make predictions for this batch
        (
            target_pcl,
            part_assingments,
            base_deformation,
            part_deformation,
            joint_params,
            intermediate_feats,
            test_canonical,
        ) = model(batch)

        (
            predicted_pcl,
            base_rot_residuum,
            canonical_base_center,
            canonical_base_scale,
            part_rot_residuum,
            part_non_axis_rot,
            can_base,
            can_part,
            R_pred,
            scale_pred,
            t_pred,
            thetas,
            pivot_points,
            rotation_axis,
            predicted_base_pcl,
        ) = model.reconstruct(
            target_pcl,
            part_assingments,
            base_deformation,
            part_deformation,
            joint_params,
        )

        log_validation(eval_log_dir, target_pcl[0].cpu().detach(), predicted_pcl[0].cpu().detach())

        # Assemble canonical
        predicted_canonical = can_base
        part_mask = part_assingments == 1
        predicted_canonical[part_mask] = can_part.squeeze(1)[part_mask]

        # Compute the loss and its gradients
        #(
        #    combined_weighted_loss,
        #    pcl_similarity_loss,
        #    base_rotation_residuum,
        #    part_non_axis_rotation_loss,
        #    zero_centered_loss,
        #    normed_scale_loss,
        #    canonical_consistency_loss,
        #) = pipeline_loss(
        #    predicted_pcl,
        #    base_rot_residuum,
        #    part_rot_residuum,
        #    part_non_axis_rot,
        #    target_pcl,
        #    canonical_base_center,
        #    canonical_base_scale,
        #    predicted_canonical,
        #    batch["cam_poses"],
        #    intermediate_feats=intermediate_feats,
        #    part_assingments=part_assingments,
        #    similarity_weight=train_cfg.similarity_weight,
        #    base_rotation_weight=train_cfg.base_rotation_weight,
        #    part_rotation_weight=train_cfg.part_rotation_weight,
        #    canonical_zero_centered_weight=train_cfg.canonical_zero_centered_weight,
        #    canonical_normed_scale_weight=train_cfg.canonical_normed_scale_weight,
        #    canonical_consistency_weight=train_cfg.canonical_consistency_weight
        #    * min(1, (epoch_index + 1) / 30),
        #)
