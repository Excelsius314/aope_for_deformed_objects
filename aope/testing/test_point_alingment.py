import open3d as o3d
import matplotlib.pyplot as plt

import torch.linalg
from torch import Tensor
import numpy as np

from scipy.spatial.transform import Rotation

import os
import sys

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(ROOT)

from modules.reconstruction.reconstruction import *

def test1():

    pcl_file = "test_data/test_pcl.ply"
    pcl: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcl_file)

    batch_size = 1
    # 3 x N
    canonical_pcl = Tensor(pcl.points)[::8].T.cuda()
    # Create rot mat that rotates around z-axis by 90 degree
    R = Tensor(
        Rotation.from_euler(seq="xyz", angles=[90, 0, 120], degrees=True).as_matrix()
    ).cuda()
    test_scale = 2
    translation = Tensor([2.4, 0, 0]).unsqueeze(0).T.cuda()

    print("Given rotation: {}".format(R))

    noise_levels = torch.linspace(0, 10, steps=20)  # adjust max noise if desired
    errors = []
    for noise_scale in noise_levels:

        freq = 5.0  # controls smoothness
        structured_noise = torch.zeros_like(canonical_pcl)

        structured_noise[0] = noise_scale * torch.sin(freq * canonical_pcl[0])
        structured_noise[1] = noise_scale * torch.sin(freq * canonical_pcl[1])
        structured_noise[2] = noise_scale * torch.sin(freq * canonical_pcl[2])

        observed_pcl = (
            (R @ canonical_pcl) * test_scale + translation  # + structured_noise
        ).unsqueeze(0)
        # observed_pcl = torch.zeros_like(canonical_pcl)
        deformation = (canonical_pcl.unsqueeze(0) - observed_pcl).requires_grad_(True)

        R_opt, t, scale, residuum, center_canonical = compute_base_pose(
            deformation.transpose(-2, -1), observed_pcl.transpose(-2, -1)
        )

        predicted_pcl = (R_opt @ canonical_pcl.unsqueeze(0)) * scale + t.transpose(
            -2, -1
        )

        prediction_error = torch.linalg.vector_norm(
            observed_pcl - predicted_pcl, dim=1
        ).sum(dim=1)

        print("Prediction error: {}".format(prediction_error))

        prediction_error.backward()

        errors.append(prediction_error.cpu().detach().numpy())

    plt.figure()
    plt.plot(noise_levels.numpy(), np.array(errors))
    plt.xlabel("Noise standard deviation")
    plt.ylabel("Total prediction error (L2 norm)")
    plt.title("Prediction error vs noise level")
    plt.grid(True)

    canonical = o3d.geometry.PointCloud()
    canonical.points = o3d.utility.Vector3dVector(canonical_pcl.transpose(-2, -1).cpu())

    observed = o3d.geometry.PointCloud()
    observed.points = o3d.utility.Vector3dVector(
        observed_pcl.transpose(-2, -1)[0].cpu()
    )

    prediction = o3d.geometry.PointCloud()
    prediction.points = o3d.utility.Vector3dVector(
        (predicted_pcl).transpose(-2, -1)[0].cpu().detach()
    )

    o3d.io.write_point_cloud("test_data/prediction.ply", prediction)
    o3d.io.write_point_cloud("test_data/observed.ply", observed)
    o3d.io.write_point_cloud("test_data/canonical.ply", canonical)

    plt.show()


def test_2():
    batch_size = 2
    part_size = 2

    pcl_file = "test_data/test_pcl.ply"
    pcl: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcl_file)

    # 3 x N
    canonical_pcl = Tensor(pcl.points)[::8].T.cuda()

    # 2 (batch) x 2 (parts) x 3 x N
    observed_pcl = (
        canonical_pcl.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, -1, -1)
    )

    pivot_point = torch.tensor([0, 0, 3.0]).unsqueeze(0).cuda()

    # batch x parts x 3 x 1
    pivot_points = (
        pivot_point.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 3)
        .transpose(-2, -1)
    )

    rotation_axis = torch.tensor([1.0, 0, 0]).cuda()
    rotation_axis = (
        rotation_axis.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 3)
    )

    theta = torch.tensor([torch.pi / 2])
    theta = theta.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 1).cuda()

    observed_pcl = apply_part_pivot_rotation(
        theta, pivot_points, rotation_axis, observed_pcl
    )

    ########## Global Pose #################

    global_R = (
        Tensor(
            Rotation.from_euler(
                seq="xyz", angles=[90, 0, 120], degrees=True
            ).as_matrix()
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 3, 3)
        .cuda()
    )

    global_s = (
        torch.Tensor([2]).unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, 1, 1).cuda()
    )
    global_t = Tensor([2.4, 0, 0]).unsqueeze(0).cuda()
    global_t = (
        global_t.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 1, 3)
        .transpose(-2, -1)
    )

    observed_pcl = apply_part_global_pose(global_t, global_s, global_R, observed_pcl)


    ############### Prediction ###############

    deformation_fields = canonical_pcl.unsqueeze(0).unsqueeze(0).expand(batch_size, part_size, -1, -1) - observed_pcl

    thetas, residuums, non_axis_rotation_residuum = compute_part_poses(observed_pcl, deformation_fields, pivot_points, rotation_axis, global_R, global_t, global_s)

    print("Thetas: {}".format(thetas))
    print("Residuums: {}".format(residuums))
    print("Non-axis rotation residuum: {}".format(non_axis_rotation_residuum))
    ############# Save to file #############

    test_base_pose_inverse = invert_base_pose(global_R, global_t, global_s, observed_pcl)

    base_pose_inverse = o3d.geometry.PointCloud()
    base_pose_inverse.points = o3d.utility.Vector3dVector(
        test_base_pose_inverse.transpose(-2, -1)[0, 0].cpu()
    )
    o3d.io.write_point_cloud("test_data/base_pose_inverse.ply", base_pose_inverse)


    observed = o3d.geometry.PointCloud()
    observed.points = o3d.utility.Vector3dVector(
        observed_pcl.transpose(-2, -1)[0, 0].cpu()
    )
    o3d.io.write_point_cloud("test_data/observed_test2.ply", observed)


if __name__ == "__main__":
    test_2()
