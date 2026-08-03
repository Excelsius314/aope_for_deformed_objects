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

pcl_dir = "/home/marek/Desktop/"
batch_size = 2

canonical_pcl = (
    Tensor(o3d.io.read_point_cloud(os.path.join(pcl_dir, "valve_canonical.ply")).points)
    .unsqueeze(0)
    .expand(batch_size, -1, -1)
)
rotated_pcl = (
    Tensor(o3d.io.read_point_cloud(os.path.join(pcl_dir, "valve_rotated.ply")).points)
    .unsqueeze(0)
    .expand(batch_size, -1, -1)
)

print(canonical_pcl.shape)

deformation_field_base = canonical_pcl - rotated_pcl

R_pred, t_pred, scale_pred, base_rot_residuum, canonical_base_center = (
    compute_base_pose(deformation_field_base, rotated_pcl)
)

print("R {}".format(R_pred))
print("t {}".format(t_pred))
print("scale {}".format(scale_pred))

predicted_pcl = apply_part_global_pose(
    t=t_pred, scale=scale_pred, R=R_pred, pcl=canonical_pcl.transpose(-1, -2)
).transpose(-1, -2)

print("Predicted pcl {}".format(predicted_pcl.shape))

pred_loss_v2_pcl = o3d.geometry.PointCloud()
pred_loss_v2_pcl.points = o3d.utility.Vector3dVector(predicted_pcl[0])
o3d.io.write_point_cloud(
    os.path.join("test_data", "predicted_v2_loss.ply"), pred_loss_v2_pcl
)

pred_loss_v2_pcl.points = o3d.utility.Vector3dVector(canonical_pcl[0])
o3d.io.write_point_cloud(
    os.path.join("test_data", "canonical_v2_loss.ply"), pred_loss_v2_pcl
)


pred_loss_v2_pcl.points = o3d.utility.Vector3dVector(rotated_pcl[0])
o3d.io.write_point_cloud(
    os.path.join("test_data", "observed_v2_loss.ply"), pred_loss_v2_pcl
)


(
    total_loss,
    pcl_similarity_loss,
    base_rotation_residuum,
    non_axis_rotation_loss,
    zero_centered_loss,
    normed_scale_loss,
    canonical_consistency,
) = pipeline_loss(
    predicted_pcl,
    base_rot_residuum,
    torch.zeros(2, 1, 1),
    torch.eye(3).expand(2, 1, 3, 3),
    rotated_pcl,
    canonical_base_center,
    scale_pred,
    canonical_pcl,
    torch.rand((2, 4, 4)),
)

print(total_loss)
print("Similarity loss {}".format(pcl_similarity_loss))
print("Canonical cosistency loss {}".format(canonical_consistency))
print("Base rot res: {}".format(base_rotation_residuum))
print("Nox axis resid: {}".format(non_axis_rotation_loss))
print("Zero centered loss : {}".format(zero_centered_loss))
print("Norms scaled loss {}".format(normed_scale_loss))