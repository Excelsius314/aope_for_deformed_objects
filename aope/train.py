from config.dataset import DatasetConfig
from config.features import (
    PCLBackboneConfig,
    PCLFeatureConfig,
    RANDLAConfig,
    VisualFeatureConfig,
    RadioConfig,
)
from config.train import TrainConfig

from utility.logging import log_prediction

from data_processing.data_loading import ProcessedAOPEDataset
from torch.utils.data import DataLoader
from torch.optim import Adagrad, AdamW

from model import AOPEModel

from modules.loss.losses import pipeline_loss

from torch.utils.tensorboard import SummaryWriter
import torch

from tqdm import tqdm
import math

import os
import datetime
import open3d as o3d

from utility.transforms import homogenous_transform, pivot_transform


def random_permute_points_shared(combined_pcl, pixel_coords, part_assignment):
    """
    Permutes the N axis using ONE shared permutation across the whole batch,
    so that index i refers to the same (now-shuffled) point identity for
    every sample. Required when later batch-averaging relies on index
    alignment across samples.

    predicted_pcl:   (B, N, 3)
    target_pcl:       (B, N, 3)
    part_assignment:  (B, N)
    """
    B, N, _ = combined_pcl.shape
    device = combined_pcl.device

    # ONE permutation, shared across the entire batch
    perm = torch.randperm(N, device=device)  # (N,)

    combined_pcl = combined_pcl[:, perm, :]
    pixel_coords = pixel_coords[:, perm, :]
    part_assignment = part_assignment[:, perm]

    return combined_pcl, pixel_coords, part_assignment


def train_one_epoch(
    epoch_index,
    model: AOPEModel,
    training_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tb_writer: SummaryWriter,
    train_cfg: TrainConfig,
    pcl_model=None,
):
    running_combined_loss = 0.0
    running_pcl_sim_loss = 0.0
    running_base_rot_residuum = 0.0
    running_part_non_axis_rotation_loss = 0.0
    running_zero_centered_loss = 0.0
    running_normed_scale_loss = 0.0
    running_canonical_consistency_loss = 0.0

    last_predicted_pcl = None
    last_target = None
    last_predicted_canonical = None
    last_cam_pose = None

    compare_canonical = None
    compare_labels = None
    last_predicted_base_pcl = None

    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in tqdm(
        enumerate(training_loader),
        total=len(training_loader),
        desc="Training on batches".format(len(training_loader)),
    ):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        pcl_shape = batch["pcls_cam_coords"].shape
        # batch["part_assingments"] = torch.zeros((pcl_shape[0], pcl_shape[1])).to("cuda:0")
        # batch["part_assingments"] [:, pcl_shape[1]//2  :] = 1

        part_assingments = torch.zeros((pcl_shape[0], pcl_shape[1]), dtype=torch.int).to("cuda:0")
        part_assingments[:, pcl_shape[1] // 2 :] = 1

        thetas = (
            (batch["joint_states"] / 360 * 2 * math.pi)
            .to(dtype=torch.float32)
            .unsqueeze(1)
        )
        # thetas = torch.zeros((2, 1, 1)).to("cuda:0").to(dtype=torch.float32)
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

        combined_pcl = torch.zeros(pcl_shape).to("cuda:0").to(dtype=torch.float32)


        combined_pcl[:, : pcl_shape[1] // 2, :] = base_pcl
        combined_pcl[:, pcl_shape[1] // 2 :, :] = part_pcl

        pixel_coords = batch["pcls_pixel_coords"]

        combined_pcl, pixel_coords, part_assingments = random_permute_points_shared(
            combined_pcl,  # (B, N, 3)
            pixel_coords,  # (B, N, 3)
            part_assingments,  # (B, N)  -- note: 2D, gather still works since expand handles it
        )

        print("combined pcl shape {}".format(combined_pcl.shape))
        print("Part assing shape {}".format(batch["part_assingments"].shape))

        batch["pcl"] = combined_pcl
        batch["part_assingments"] = part_assingments
        batch["pcls_pixel_coords"] = pixel_coords

        part_mask = batch["part_assingments"] == 1


        # batch["pcl"] = homogenous_transform(batch["obj_poses"], pcl_model.expand((pcl_shape[0], pcl_shape[1], 3)))

        # Make predictions for this batch
        (
            target_pcl,
            part_assingments,
            base_deformation,
            part_deformation,
            joint_params,
            intermediate_feats_loss,
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

        # Assemble canonical
        predicted_canonical = can_base
        part_mask = part_assingments == 1
        predicted_canonical[part_mask] = can_part.squeeze(1)[part_mask]

        # Compute the loss and its gradients
        (
            combined_weighted_loss,
            pcl_similarity_loss,
            base_rotation_residuum,
            part_non_axis_rotation_loss,
            zero_centered_loss,
            normed_scale_loss,
            canonical_consistency_loss,
        ) = pipeline_loss(
            predicted_pcl,
            base_rot_residuum,
            part_rot_residuum,
            part_non_axis_rot,
            target_pcl,
            canonical_base_center,
            canonical_base_scale,
            predicted_canonical,
            batch["cam_poses"],
            intermediate_feats=intermediate_feats_loss,
            part_assingments=part_assingments,
            similarity_weight=train_cfg.similarity_weight,
            base_rotation_weight=train_cfg.base_rotation_weight,
            part_rotation_weight=train_cfg.part_rotation_weight,
            canonical_zero_centered_weight=train_cfg.canonical_zero_centered_weight,
            canonical_normed_scale_weight=train_cfg.canonical_normed_scale_weight,
            canonical_consistency_weight=train_cfg.canonical_consistency_weight
            #* min(1, (epoch_index + 1) / 30),
        )

        combined_weighted_loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_combined_loss += combined_weighted_loss.item()
        running_pcl_sim_loss += pcl_similarity_loss.mean().item()
        running_base_rot_residuum += base_rotation_residuum.mean().item()
        running_part_non_axis_rotation_loss += part_non_axis_rotation_loss.mean().item()
        running_zero_centered_loss += zero_centered_loss.mean().item()
        running_normed_scale_loss += normed_scale_loss.mean().item()
        running_canonical_consistency_loss += canonical_consistency_loss.item()

        for f, data_idx in enumerate(batch["data_idx"]):
            if data_idx == train_cfg.compare_idx:
                compare_canonical = predicted_canonical[f].cpu().detach()
                compare_labels = {
                    "R": R_pred[f],
                    "s": scale_pred[f],
                    "t": t_pred[f],
                    "theta": thetas[f],
                    "pivots": pivot_points[f],
                    "rot_axis": rotation_axis[f],
                }

        if i == len(training_loader) - 1:  # % train_cfg.log_frequency == 0:
            tb_x = epoch_index * len(training_loader) + i + 1

            tb_writer.add_scalar(
                "Loss/train", running_combined_loss / train_cfg.log_frequency, tb_x
            )
            tb_writer.add_scalar(
                "PCL-Similarity-Loss/train",
                running_pcl_sim_loss / train_cfg.log_frequency,
                tb_x,
            )
            tb_writer.add_scalar(
                "Base-Rot-Residuum/train",
                running_base_rot_residuum / train_cfg.log_frequency,
                tb_x,
            )
            tb_writer.add_scalar(
                "Part-Non-Axis-Rot-Loss/train",
                running_part_non_axis_rotation_loss / train_cfg.log_frequency,
                tb_x,
            )
            tb_writer.add_scalar(
                "Zero-Centered-Loss/train",
                running_zero_centered_loss / train_cfg.log_frequency,
                tb_x,
            )
            tb_writer.add_scalar(
                "Normed-Scale-Loss/train",
                running_normed_scale_loss / train_cfg.log_frequency,
                tb_x,
            )
            tb_writer.add_scalar(
                "Canonical-Consistency/train",
                running_canonical_consistency_loss / train_cfg.log_frequency,
                tb_x,
            )

            running_combined_loss = 0.0
            running_pcl_sim_loss = 0.0
            running_base_rot_residuum = 0.0
            running_part_non_axis_rotation_loss = 0.0
            running_zero_centered_loss = 0.0
            running_normed_scale_loss = 0.0
            running_canonical_consistency_loss = 0.0

            tb_writer.flush()

        if i == len(training_loader) - 1:
            last_target = target_pcl[0].cpu().detach()
            last_predicted_pcl = predicted_pcl[0].cpu().detach()
            last_predicted_canonical = predicted_canonical[0].cpu().detach()
            last_cam_pose = batch["cam_poses"][0].cpu().detach()
            last_predicted_base_pcl = predicted_base_pcl[0].cpu().detach()

    return (
        last_loss,
        last_target,
        last_predicted_pcl,
        last_predicted_canonical,
        compare_canonical,
        compare_labels,
        last_cam_pose,
        last_predicted_base_pcl,
        test_canonical,
    )


def setup_run_dir(train_conf):

    dataset_dir = train_conf.run_name
    # Check if previous runs of preprocessed were saved
    idx = len(
        [
            dir_name
            for dir_name in os.listdir(train_conf.runs_dir)
            if dir_name.startswith(dataset_dir)
        ]
    )
    if idx > 0:
        dataset_dir = dataset_dir + "_v{}".format(idx)

    dt = datetime.datetime.now()
    dataset_dir = dataset_dir + "_" + dt.strftime("%b_%d_%H_%M")

    dataset_dir = os.path.join(train_conf.runs_dir, dataset_dir)
    log_dir = os.path.join(dataset_dir, "logs")
    weights_dir = os.path.join(dataset_dir, "weights")

    os.makedirs(dataset_dir)
    os.makedirs(log_dir)
    os.makedirs(weights_dir)

    return dataset_dir, log_dir, weights_dir


def train():
    dataset_conf = DatasetConfig()
    dataset_conf.path = (
        "/home/marek/aope_for_deformed_objects/aope/data/valve_full_preprocessed_5"
    )
    dataset_conf.pcl_size = 512

    train_conf = TrainConfig()
    train_conf.run_name = "consistency"
    train_conf.batch_size = 30
    train_conf.num_epochs = 500

    visual_feature_conf = VisualFeatureConfig()
    visual_feature_conf.backbone_config = RadioConfig()
    pcl_feat_conf = PCLFeatureConfig()
    pcl_feat_conf.backbone_config = RANDLAConfig()

    ########### Setup run dir ###################

    run_dir, log_dir, weights_dir = setup_run_dir(train_conf)

    ############## Load Dataset #####################
    device = "cuda:0"

    dataset = ProcessedAOPEDataset(dataset_conf, device)
    loader = DataLoader(
        dataset,
        batch_size=train_conf.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model = AOPEModel(dataset_conf, visual_feature_conf, pcl_feat_conf, device).to(
        device
    )

    model.load_state_dict(torch.load("/home/marek/aope_for_deformed_objects/aope/runs/consistency_v1_Jun_22_06_37/weights/weights_epoch_136", weights_only=True))
    

    # Full observation training
    base_model = o3d.io.read_point_cloud(
        "/home/marek/aope_for_deformed_objects/aope/data/valve_base.ply"
    ).farthest_point_down_sample(dataset_conf.pcl_size // 2)
    base_model = torch.tensor(base_model.points).to("cuda:0").to(dtype=torch.float32)
    part_model = o3d.io.read_point_cloud(
        "/home/marek/aope_for_deformed_objects/aope/data/valve_handle.ply"
    ).farthest_point_down_sample(dataset_conf.pcl_size // 2)
    part_model = torch.tensor(part_model.points).to("cuda:0").to(dtype=torch.float32)
    pcl_model = {"base": base_model, "part": part_model}

    # optimizer = Adagrad(model.parameters())
    optimizer = AdamW(model.parameters(), lr=0.001)
    tb_writer = SummaryWriter(run_dir)

    for epoch in tqdm(
        range(train_conf.num_epochs),
        desc="Training run ({} epochs)".format(train_conf.num_epochs),
    ):
        (
            last_loss,
            last_target,
            last_predicted_pcl,
            last_predicted_canonical,
            compare_canonical,
            compare_labels,
            last_cam_pose,
            last_predicted_base_pcl,
            test_canonical,
        ) = train_one_epoch(
            epoch, model, loader, optimizer, tb_writer, train_conf, pcl_model
        )

        log_prediction(
            log_dir,
            epoch,
            last_target,
            last_predicted_pcl,
            last_predicted_canonical,
            compare_canonical,
            last_cam_pose,
            test_canonical,
        )

        torch.save(
            model.state_dict(),
            os.path.join(weights_dir, "weights_epoch_{}".format(epoch)),
        )


if __name__ == "__main__":

    train()
