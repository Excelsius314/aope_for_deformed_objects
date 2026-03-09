import open3d as o3d
import matplotlib.pyplot as plt

from torch import Tensor
import numpy as np

from scipy.spatial.transform import Rotation

import os
import sys

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(ROOT)

from modules.reconstruction.reconstruction import *
from modules.loss.losses import *


def init_base_params(
    rot_angles_val,
    translation_val,
    scale_val,
    batch_size,
    test_pcl: o3d.geometry.PointCloud,
):

    canonical_pcl = (
        Tensor(test_pcl.points)[::8].cuda().unsqueeze(0).expand(batch_size, -1, -1)
    )

    print(canonical_pcl.shape)

    # Make zero centered
    canonical_pcl = canonical_pcl - canonical_pcl.mean(dim=1, keepdim=True)

    global_R = (
        Tensor(
            Rotation.from_euler(
                seq="xyz", angles=rot_angles_val, degrees=True
            ).as_matrix()
        )
        .unsqueeze(0)
        .expand(batch_size, 3, 3)
        .cuda()
    )

    global_s = torch.Tensor(scale_val).unsqueeze(0).expand(batch_size, 1, 1).cuda()
    global_t = Tensor(translation_val).unsqueeze(0).expand(batch_size, 1, 3).cuda()

    return canonical_pcl, global_R, global_s, global_t


def init_part_params(
    pivot_point_val,
    rotation_axis_val,
    theta_val,
    global_angle_vals,
    global_t_val,
    global_s_val,
    batch_size,
    part_size,
    test_pcl: o3d.geometry.PointCloud,
):

    canonical_pcl = (
        Tensor(test_pcl.points)[::8]
        .cuda()
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, -1, -1)
    )

    pivot_point = torch.tensor(pivot_point_val, device=canonical_pcl.device)
    # batch x parts x 3 x 1
    pivot_points = (
        pivot_point.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 3)
        .transpose(-2, -1)
    )
    rotation_axis = torch.tensor(rotation_axis_val).cuda()
    rotation_axis = (
        rotation_axis.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 3)
    )

    theta = torch.tensor(theta_val)
    theta = theta.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 1).cuda()

    global_R = (
        Tensor(
            Rotation.from_euler(
                seq="xyz", angles=global_angle_vals, degrees=True
            ).as_matrix()
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 3, 3)
        .cuda()
    )

    global_s = (
        torch.Tensor(global_s_val)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 1)
        .cuda()
    )
    global_t = (
        Tensor(global_t_val)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 3)
        .transpose(-2, -1)
        .cuda()
    )

    return (
        canonical_pcl,
        pivot_points,
        rotation_axis,
        theta,
        global_R,
        global_t,
        global_s,
    )


def init_combined_params(
    pivot_point_val,
    rotation_axis_val,
    theta_val,
    global_angle_vals,
    global_t_val,
    global_s_val,
    batch_size,
    part_size,
    test_pcl: o3d.geometry.PointCloud,
):

    obj_pcl = Tensor(test_pcl.points)[::8].cuda()

    obj_pcl = obj_pcl - obj_pcl.mean(dim=0, keepdim=True)

    base_pcl = torch.clone(obj_pcl).unsqueeze(0).expand(batch_size, -1, -1)
    part_pcl = (
        torch.clone(obj_pcl + torch.tensor([0.0, 0.0, 1.0]).expand(1, 3).cuda())
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, -1, -1)
    )

    pivot_point = torch.tensor(pivot_point_val, device=obj_pcl.device)
    # batch x parts x 3 x 1
    pivot_points = (
        pivot_point.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 3)
        .transpose(-2, -1)
    )
    rotation_axis = torch.tensor(rotation_axis_val).cuda()
    rotation_axis = (
        rotation_axis.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 3)
    )

    theta = torch.tensor(theta_val)
    theta = theta.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 1).cuda()

    global_R = (
        Tensor(
            Rotation.from_euler(
                seq="xyz", angles=global_angle_vals, degrees=True
            ).as_matrix()
        )
        .unsqueeze(0)
        .expand(batch_size, 3, 3)
        .cuda()
    )

    global_s = torch.Tensor(global_s_val).unsqueeze(0).expand(batch_size, 1, 1).cuda()
    global_t = Tensor(global_t_val).unsqueeze(0).expand(batch_size, 1, 3).cuda()

    return (
        base_pcl,
        part_pcl,
        pivot_points,
        rotation_axis,
        theta,
        global_R,
        global_t,
        global_s,
    )


######################## Test #############################


def part_test():
    batch_size = 2
    part_size = 2

    pcl_file = "test_data/test_pcl.ply"
    pcl: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcl_file)

    canonical_pcl, pivot_points, rotation_axis, theta, global_R, global_t, global_s = (
        init_part_params(
            [0, 0, 3.0],
            [1.0, 0, 0],
            [torch.pi / 2],
            [90, 0, 120],
            [2.4, 0, 0],
            [2],
            batch_size,
            part_size,
            pcl,
        )
    )

    ############### Transform ###############

    # batch_size x part_size x N x 3
    observed_pcl = torch.clone(canonical_pcl)

    print("Rotating thetas: {}".format(theta.shape))
    observed_pcl = apply_part_pivot_rotation(
        theta, pivot_points, rotation_axis, observed_pcl.transpose(-1, -2)
    ).transpose(-1, -2)

    observed_pcl = apply_part_global_pose(
        global_t, global_s, global_R, observed_pcl.transpose(-2, -1)
    ).transpose(-2, -1)

    deformation_fields = (canonical_pcl - observed_pcl).requires_grad_(True)

    ############### Prediction ###############

    thetas, residuums, non_axis_rotation_residuum = compute_part_poses(
        observed_pcl,
        deformation_fields,
        pivot_points.transpose(-2, -1),
        rotation_axis,
        global_R,
        global_t,
        global_s,
    )

    predicted_pcl = apply_part_pivot_rotation(
        theta=thetas,
        pivot_point=pivot_points,
        rotation_axis=rotation_axis,
        pcl=canonical_pcl.transpose(-2, -1),
    ).transpose(-1, -2)
    print(predicted_pcl.shape)

    predicted_pcl = apply_part_global_pose(
        t=global_t, scale=global_s, R=global_R, pcl=predicted_pcl.transpose(-1, -2)
    ).transpose(-1, -2)

    residuum = torch.zeros((batch_size)).cuda()
    center_canonical = torch.zeros((batch_size, 3, 1)).cuda()

    ## "combine parts"
    predicted_pcl = torch.cat((predicted_pcl[:, 0], predicted_pcl[:, 1]), dim=1)
    print(predicted_pcl.shape)
    observed_pcl = torch.cat((observed_pcl[:, 0], observed_pcl[:, 1]), dim=1)

    loss = pipeline_loss(
        predicted_pcl,
        residuum,
        non_axis_rotation_residuum,
        observed_pcl,
        center_canonical,
    )

    loss.backward()


def base_test():
    batch_size = 2

    pcl_file = "test_data/test_pcl.ply"
    pcl: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcl_file)

    canonical_pcl, global_R, global_s, global_t = init_base_params(
        [90, 0, 120], [2.4, 0, 0], [2], batch_size, pcl
    )

    ############### Transform ###############

    # 2 (batch) x N x 3
    observed_pcl = torch.clone(canonical_pcl)

    observed_pcl = apply_part_global_pose(
        global_t.transpose(-2, -1), global_s, global_R, observed_pcl.transpose(-2, -1)
    ).transpose(-2, -1)

    deformation_fields = (canonical_pcl - observed_pcl).requires_grad_(True)

    print("Deformation fields shape: ", deformation_fields.shape)
    print("Observed pcl shape: ", observed_pcl.shape)

    ############### Prediction ###############

    R_pred, t_pred, scale_pred, residuum, center_canonical = compute_base_pose(
        deformation_fields, observed_pcl
    )

    predicted_pcl = apply_part_global_pose(
        t=t_pred.transpose(-2, -1),
        scale=scale_pred,
        R=R_pred,
        pcl=canonical_pcl.transpose(-2, -1),
    ).transpose(-2, -1)

    print("Predicted Rotation: {}".format(R_pred.round(decimals=1)))

    ############### Loss ###############

    non_axis_rotation_residuum = (
        torch.eye(3, device=canonical_pcl.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 2, 3, 3)
    )

    loss = pipeline_loss(
        predicted_pcl=predicted_pcl,
        base_rotation_residuum=residuum,
        part_rotation_residuums=non_axis_rotation_residuum,
        target_pcl=observed_pcl,
        center_canonical=center_canonical,
    )

    loss.backward()


def combined_test():

    batch_size = 2
    part_size = 2

    pcl_file = "test_data/test_pcl.ply"
    pcl: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcl_file)

    (
        base_pcl,
        part_pcls,
        pivot_points,
        rotation_axis,
        theta,
        global_R,
        global_t,
        global_s,
    ) = init_combined_params(
        [0, 0, 3.0],
        [1.0, 0, 0],
        [torch.pi / 2],
        [90, 0, 120],
        [2.4, 0, 0],
        [2],
        batch_size,
        part_size,
        pcl,
    )

    ############### Transform ###############

    # batch_size x part_size x N x 3
    observed_part_pcls = torch.clone(part_pcls)
    observed_base_pcl = torch.clone(base_pcl)

    observed_part_pcls = apply_part_pivot_rotation(
        theta, pivot_points, rotation_axis, observed_part_pcls.transpose(-1, -2)
    ).transpose(-1, -2)

    observed_part_pcls = apply_part_global_pose(
        global_t, global_s, global_R, observed_part_pcls.transpose(-2, -1)
    ).transpose(-2, -1)

    observed_base_pcl = apply_part_global_pose(
        global_t, global_s, global_R, observed_base_pcl.transpose(-2, -1)
    ).transpose(-2, -1)

    target_pcl = torch.cat(
        [
            observed_base_pcl,
            observed_part_pcls.reshape(observed_part_pcls.shape[0], -1, 3),
        ],
        dim=-2,
    )

    noise_levels = torch.linspace(0, 10, steps=20)  # adjust max noise if desired

    for noise_scale in noise_levels:

        freq = 1.0  # controls smoothness
        structured_noise_base = torch.zeros_like(base_pcl)
        structured_noise_part = torch.zeros_like(part_pcls)

        structured_noise_base[:, :, 0] = noise_scale * torch.sin(freq * base_pcl[:, :, 0])
        structured_noise_base[:, :, 1] = noise_scale * torch.sin(freq * base_pcl[:, :, 1])
        structured_noise_base[:, :, 2] = noise_scale * torch.sin(freq * base_pcl[:, :, 2])

        deformation_field_base = (base_pcl - observed_base_pcl).requires_grad_(True) + structured_noise_base
        deformation_fields_parts = (part_pcls - observed_part_pcls).requires_grad_(True)

        ############### Prediction ###############

        predicted_pcl, base_rot_residuum, canonical_base_center,  part_rot_residuum, part_non_axis_rot,  = reconstruct_pose_params(
            deformation_field_base,
            observed_base_pcl,
            deformation_fields_parts,
            observed_part_pcls,
            pivot_points,
            rotation_axis,
        )

        ########################### Loss ##################

     

        loss = pipeline_loss(
            predicted_pcl,
            base_rot_residuum + part_rot_residuum,
            part_non_axis_rot,
            target_pcl,
            canonical_base_center,
        )

        loss.backward()


if __name__ == "__main__":
    combined_test()
