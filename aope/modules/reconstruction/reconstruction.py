from torch import Tensor
import torch


def apply_part_global_pose(t: Tensor, scale: Tensor, R: Tensor, pcl: Tensor):
    # pcl : batch x part_size x 3 x N or batch x 3 x N
    assert pcl.shape[-2] == 3
    assert t.shape[-2] == 1  # singleton dimension
    assert t.shape[-1] == 3

    center = pcl.mean(dim=-1, keepdim=True)
    pcl = (pcl - center) * scale + center

    # out: batch (x patch_size) x 3 x N
    return (R @ pcl) + t.transpose(-2, -1)


def apply_part_pivot_rotation(
    theta: Tensor, pivot_point: Tensor, rotation_axis: Tensor, pcl: Tensor
):
    batch_size, part_size = pcl.shape[0], pcl.shape[1]

    assert rotation_axis.shape == (batch_size, part_size, 3)
    assert pivot_point.shape == (batch_size, part_size, 3, 1)
    assert pcl.shape == (batch_size, part_size, 3, pcl.shape[-1])
    assert theta.shape == (batch_size, part_size, 1)

    rotation_axis = rotation_axis / torch.linalg.vector_norm(
        rotation_axis, dim=-1, keepdim=True
    )

    ux, uy, uz = rotation_axis.unbind(-1)
    O = torch.zeros_like(ux)

    K = torch.stack(
        [
            torch.stack([O, -uz, uy], dim=-1),
            torch.stack([uz, O, -ux], dim=-1),
            torch.stack([-uy, ux, O], dim=-1),
        ],
        dim=-2,
    )

    # Rodrigues
    R = (
        torch.eye(3, device=theta.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, part_size, 3, 3)
        + torch.sin(theta).unsqueeze(-1) * K
        + (1 - torch.cos(theta).unsqueeze(-1)) * torch.matmul(K, K)
    )
    assert R.shape == (batch_size, part_size, 3, 3)

    return (R @ (pcl - pivot_point)) + pivot_point


def assemble_point_cloud(
    canonical_base_pcl: Tensor,
    base_R: Tensor,
    base_s: Tensor,
    base_t,
    canonical_part_pcls: Tensor,
    part_thetas: Tensor,
    pivot_points: Tensor,
    rotation_axis: Tensor,
):
    predicted_base_pcl = apply_part_global_pose(
        t=base_t, scale=base_s, R=base_R, pcl=canonical_base_pcl.transpose(-1, -2)
    ).transpose(-1, -2)

    predicted_part_pcl = apply_part_pivot_rotation(
        theta=part_thetas,
        pivot_point=pivot_points,
        rotation_axis=rotation_axis,
        pcl=canonical_part_pcls.transpose(-2, -1),
    ).transpose(-1, -2)

    predicted_part_pcl = (
        apply_part_global_pose(
            t=base_t, scale=base_s, R=base_R, pcl=predicted_part_pcl.transpose(-1, -2)
        )
        .transpose(-1, -2)
        .reshape(predicted_base_pcl.shape[0], -1, 3)
    )

    return torch.cat([predicted_base_pcl, predicted_part_pcl], dim=-2)


def kabsch_umeyama(canonical_pc: Tensor, observed_pc: Tensor):
    H = (observed_pc.transpose(-2, -1) @ canonical_pc) / canonical_pc.shape[-2]

    print(H.shape)
    U, D, Vt = torch.linalg.svd(H)

    assert H.shape[-2] == 3
    assert H.shape[-1] == 3

    d = torch.sign(torch.det(U) * torch.det(Vt))

    if len(canonical_pc.shape) == 2:
        # For non-batched inputs
        S = torch.diag(torch.tensor([1, 1, d], device=canonical_pc.device))
    elif len(canonical_pc.shape) == 3:
        # For base part where canonical_pcl.shape = batch_size x N x 3
        S = (
            torch.eye(3, device=canonical_pc.device)
            .unsqueeze(0)
            .repeat(canonical_pc.shape[0], 1, 1)
        )
        S[:, 2, 2] = d

    else:
        # For part pcs where canonical_pcl.shape = batch_size x part_size x N x 3
        S = (
            torch.eye(3, device=canonical_pc.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(canonical_pc.shape[0], canonical_pc.shape[1], 1, 1)
        )
        S[:, :, 2, 2] = d

    R = U @ S @ Vt

    residuum = torch.linalg.vector_norm(
        observed_pc - (R @ canonical_pc.transpose(-2, -1)).transpose(-2, -1), dim=-1
    ).sum(dim=-1)

    return R, residuum


def optimal_angle_about_axis(R: Tensor, rotation_axis: Tensor):
    """
    R : batch_size X part_size X 3 X 3 rotation matrix
    rotation_axis: batch_size X part_size X 3 rotation_axis
    u : 3D unit axis vector
    Returns optimal theta (radians)
    """
    rotation_axis = rotation_axis / torch.linalg.vector_norm(
        rotation_axis, dim=-1, keepdim=True
    )
    ux, uy, uz = rotation_axis.unbind(-1)
    rotation_axis = rotation_axis.unsqueeze(-1).transpose(-1, -2)

    O = torch.zeros_like(ux)
    K = torch.stack(
        [
            torch.stack([O, -uz, uy], dim=-1),
            torch.stack([uz, O, -ux], dim=-1),
            torch.stack([-uy, ux, O], dim=-1),
        ],
        dim=-2,
    )

    a = R.diagonal(dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(
        -1
    ) - rotation_axis @ R @ rotation_axis.transpose(-1, -2)
    b = (K @ R).diagonal(dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    theta = -torch.atan2(b, a)
    I = (
        torch.eye(3, device=R.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(R.shape[0], R.shape[1], 3, 3)
    )

    residual_rotation = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    R_res = residual_rotation.transpose(-2, -1) @ R

    return theta.squeeze(-1), R_res


def kabsch_umeyame_pivot_rotation(
    canonical_pc: Tensor,
    observed_pc: Tensor,
    pivot_points: Tensor,
    rotation_axis: Tensor,
):
    # O, C : batch_size x part_n X 3 X N
    # pivot_points : batch_size X part_n X 3 X 1
    observed_pc -= pivot_points
    canonical_pc -= pivot_points

    R, rotation_residuum = kabsch_umeyama(
        canonical_pc.transpose(-2, -1),
        observed_pc.transpose(-2, -1),
    )

    theta, R_res = optimal_angle_about_axis(R, rotation_axis)

    return theta, rotation_residuum, R_res


def normalize_pcl(pcl: Tensor):

    # Batch x N x 3 or B x Part size X N x 3
    assert pcl.shape[-1] == 3

    if len(pcl.shape) >= 3:
        # Pcl shape Batch_SizexNx3
        center = pcl.mean(dim=-2, keepdim=True)
        scale = torch.sqrt(
            (pcl - center).pow(2).sum(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        )
        return (pcl - center) / scale, center, scale
    else:
        # Not batched
        center = pcl.mean(dim=0)
        scale = torch.sqrt((pcl - center).pow(2).sum(dim=1).mean())
        return (pcl - center) / scale, center, scale


def compute_base_pose(deformation_field: Tensor, observed_pc: Tensor):
    """
    deformation_field : B x N x 3
    observed_pcs : B X N x 3
    """
    assert deformation_field.shape == observed_pc.shape
    assert deformation_field.shape[-1] == 3

    canonical_pc = observed_pc + deformation_field

    normalized_observed, center_observed, scale_observed = normalize_pcl(observed_pc)
    normalized_canonical, center_canonical, scale_canonical = normalize_pcl(
        canonical_pc
    )

    R, residuum = kabsch_umeyama(normalized_canonical, normalized_observed)

    t = center_observed - (R @ center_canonical.transpose(-2, -1)).transpose(-2, -1) * (
        scale_observed / scale_canonical
    )

    scale = scale_observed / scale_canonical

    return R, t, scale, residuum, center_canonical


def invert_base_pose(R: Tensor, t: Tensor, scale: Tensor, pcl: Tensor):
    """
    Invert the base pose transformation.
    R: batch_size x part_size x 3 x 3
    t: batch_size x part_size x 3 x 1
    scale: batch_size x part_size x 1 x 1
    observed_pcl: batch_size x part_size x 3 x N
    """
    center = pcl.mean(dim=-1, keepdim=True)
    pcl = (pcl - center) * (1 / scale) + center
    return R.transpose(-2, -1) @ (pcl - t)


def compute_part_poses(
    observed_part_pcs: Tensor,
    deformation_fields: Tensor,
    pivot_points: Tensor,
    rotation_axis: Tensor,
    base_R: Tensor,
    base_t: Tensor,
    base_scale: Tensor,
):
    assert observed_part_pcs.dim() == 4
    assert observed_part_pcs.shape[-1] == 3
    assert deformation_fields.dim() == 4
    assert deformation_fields.shape[-1] == 3
    assert pivot_points.shape[-2] == 1  # singleton dimension
    assert base_t.shape[-2] == 1
    assert base_t.shape[-1] == 3  # singleton dimension
    assert base_t.dim() == 3

    # Assume deformation fields translate to pose-invariant space, i.e. object model space
    canonical_part_pcs = observed_part_pcs + deformation_fields

    # Eliminate effect of base part pose, i.e. global 6D pose
    observed_part_pcs = invert_base_pose(
        base_R.unsqueeze(1).expand(
            observed_part_pcs.shape[0], observed_part_pcs.shape[1], 3, 3
        ),
        base_t.unsqueeze(1)
        .expand(observed_part_pcs.shape[0], observed_part_pcs.shape[1], 1, 3)
        .transpose(-2, -1),
        base_scale.unsqueeze(1).expand(
            observed_part_pcs.shape[0], observed_part_pcs.shape[1], 1, 1
        ),
        observed_part_pcs.transpose(-1, -2),
    ).transpose(-1, -2)

    assert observed_part_pcs.shape[-1] == 3

    return kabsch_umeyame_pivot_rotation(
        canonical_part_pcs.transpose(-1, -2),
        observed_part_pcs.transpose(-1, -2),
        pivot_points.transpose(-1, -2),
        rotation_axis,
    )


def reconstruct_pose_params(
    deformation_field_base: Tensor,
    observed_base_pcl: Tensor,
    deformation_fields_parts: Tensor,
    observed_part_pcls: Tensor,
    pivot_points: Tensor,
    rotation_axis: Tensor,
):

    R_pred, t_pred, scale_pred, base_rot_residuum, canonical_base_center = compute_base_pose(
        deformation_field_base, observed_base_pcl
    )

    thetas, part_rot_residuum, part_non_axis_rot = compute_part_poses(
        observed_part_pcls,
        deformation_fields_parts,
        pivot_points.transpose(-2, -1),
        rotation_axis,
        R_pred,
        t_pred,
        scale_pred,
    )

    predicted_pcl = assemble_point_cloud(
        observed_base_pcl + deformation_field_base,
        R_pred,
        scale_pred,
        t_pred,
        observed_part_pcls + deformation_fields_parts,
        thetas,
        pivot_points,
        rotation_axis,
    )

    return predicted_pcl, base_rot_residuum, canonical_base_center, part_rot_residuum.sum(dim=1), part_non_axis_rot
