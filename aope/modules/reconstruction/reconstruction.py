from torch import Tensor
import torch

from utility.math import farthest_point_sampling


def pad_non_part_points(
    canonical_pcl: Tensor,
    observed_pcl: Tensor,
    part_assingments: Tensor,
    part_id: int,
    strategy=None,
):

    B = canonical_pcl.shape[0]
    out_canonical = canonical_pcl.clone()
    out_observed = observed_pcl.clone()

    for b in range(B):
        part_mask = part_assingments[b] == part_id  # (N,) bool
        part_indices = part_mask.nonzero(as_tuple=True)[0]  # (M_k,)
        non_part_indices = (~part_mask).nonzero(as_tuple=True)[0]  # (N - M_k,)

        if part_indices.shape[0] == 0:
            # No points for part k in this batch item — nothing to sample from
            continue

        if strategy == "Random-Sample":

            # Sample with replacement from part-k indices to fill non-part positions
            sampled = part_indices[
                torch.randint(0, part_indices.shape[0], (non_part_indices.shape[0],))
            ]  # (N - M_k,)

            out_canonical[b, non_part_indices] = canonical_pcl[b, sampled]
            out_observed[b, non_part_indices] = observed_pcl[b, sampled]

        if strategy == "Zero-Padding":

            out_canonical[b, non_part_indices] = 0
            out_observed[b, non_part_indices] = 0

    return out_canonical, out_observed


def apply_padding(
    deformation_field: Tensor,
    observed_pcl: Tensor,
    canonical_pcl,
    is_part,
    padding_size,
    strategy=None,
):
    if not strategy:
        return deformation_field, observed_pcl

    batch_size = deformation_field.shape[0]

    # if padding_size == 0:
    #    return deformation_field, observed_pcl

    if strategy == "Random-Sample":

        if is_part:
            padding_dim = 2
            indices = (
                torch.rand((batch_size, observed_pcl.shape[1], padding_size))
                * observed_pcl.shape[2]
            ).to(dtype=torch.int64)
            batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
            part_indices = torch.arange(observed_pcl.shape[1]).unsqueeze(
                1
            )  # (part_size, 1)

            padding_canonical = canonical_pcl[
                batch_indices, part_indices, indices
            ]  # (batch_size, padding_dim, embedding_dim)
            deformation_field = torch.concat(
                (
                    deformation_field,
                    padding_canonical
                    - observed_pcl[batch_indices, part_indices, indices],
                ),
                dim=padding_dim,
            )
            observed_pcl = torch.concat(
                (observed_pcl, observed_pcl[batch_indices, part_indices, indices]),
                dim=padding_dim,
            )
        else:
            padding_dim = 1
            indices = (
                torch.rand((batch_size, padding_size)) * observed_pcl.shape[1]
            ).to(dtype=torch.int64)
            batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)

            padding_canonical = canonical_pcl[
                batch_indices, indices
            ]  # (batch_size, padding_dim, embedding_dim)
            deformation_field = torch.concat(
                (
                    deformation_field,
                    padding_canonical - observed_pcl[batch_indices, indices],
                ),
                dim=padding_dim,
            )
            observed_pcl = torch.concat(
                (observed_pcl, observed_pcl[batch_indices, indices]), dim=padding_dim
            )

        return deformation_field, observed_pcl

    if strategy == "Zero-Padding":

        if is_part:
            part_size = observed_pcl.shape[1]
            padding = torch.zeros((batch_size, part_size, padding_size, 3)).to(
                observed_pcl.device
            )
            deformation_field = torch.concat((deformation_field, padding), dim=2)
            observed_pcl = torch.concat((observed_pcl, padding), dim=2)
        else:
            padding = torch.zeros((batch_size, padding_size, 3)).to(observed_pcl.device)
            deformation_field = torch.concat((deformation_field, padding), dim=1)
            observed_pcl = torch.concat((observed_pcl, padding), dim=1)

        return deformation_field, observed_pcl

    if strategy == "Farthest-Point-Sampling":

        if is_part:
            padding_dim = 2
            part_size = observed_pcl.shape[1]

            part_point_indices = []
            for batch in range(batch_size):
                part_point_indices.append(
                    farthest_point_sampling(
                        canonical_pcl[batch], padding_size
                    ).unsqueeze(0)
                )
            part_point_indices = torch.concat(part_point_indices)

            batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
            part_indices = torch.arange(observed_pcl.shape[1]).unsqueeze(
                1
            )  # (part_size, 1)
            padding_canonical = canonical_pcl[
                batch_indices, part_indices, part_point_indices
            ]
            deformation_field = torch.concat(
                (
                    deformation_field,
                    padding_canonical
                    - observed_pcl[batch_indices, part_indices, part_point_indices],
                ),
                dim=padding_dim,
            )
            observed_pcl = torch.concat(
                (
                    observed_pcl,
                    observed_pcl[batch_indices, part_indices, part_point_indices],
                ),
                dim=padding_dim,
            )
        else:
            padding_dim = 1
            indices = farthest_point_sampling(canonical_pcl, padding_size)
            batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)

            padding_canonical = canonical_pcl[
                batch_indices, indices
            ]  # (batch_size, padding_dim, embedding_dim)
            deformation_field = torch.concat(
                (
                    deformation_field,
                    padding_canonical - observed_pcl[batch_indices, indices],
                ),
                dim=padding_dim,
            )
            observed_pcl = torch.concat(
                (observed_pcl, observed_pcl[batch_indices, indices]), dim=padding_dim
            )

        return deformation_field, observed_pcl

    return deformation_field, observed_pcl


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
    part_assingment: Tensor,
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
            t=base_t.unsqueeze(1),
            scale=base_s.unsqueeze(1),
            R=base_R.unsqueeze(1),
            pcl=predicted_part_pcl.transpose(-1, -2),
        ).transpose(-1, -2)
        # .reshape(predicted_base_pcl.shape[0], -1, 3)
    )

    assembled_pcl = torch.zeros_like(canonical_base_pcl)

    K = predicted_part_pcl.shape[1] + 1
    for k in range(K):
        mask = part_assingment == k

        if k == 0:
        #   assembled_pcl = predicted_base_pcl
            assembled_pcl[mask] = predicted_base_pcl[mask]
        else:
            assembled_pcl[mask] = predicted_part_pcl[:, k - 1][mask]

    return assembled_pcl, predicted_base_pcl


def kabsch_umeyama(canonical_pc: Tensor, observed_pc: Tensor):
    H = (observed_pc.transpose(-2, -1) @ canonical_pc) / canonical_pc.shape[-2]

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


def compute_base_pose(canonical_pcl, observed_pcl):
    """
    deformation_field : B x N x 3
    observed_pcs : B X N x 3
    """

    normalized_observed, center_observed, scale_observed = normalize_pcl(observed_pcl)
    normalized_canonical, center_canonical, scale_canonical = normalize_pcl(
        canonical_pcl
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
    canonical_part_pcs = deformation_fields #observed_part_pcs + deformation_fields

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
    part_assingment: Tensor,
    padding_size=0,
):

    N = observed_base_pcl.shape[1]
    B = observed_base_pcl.shape[0]
    can_base = deformation_field_base#observed_base_pcl + deformation_field_base
    #can_base = (
    #    can_base_hypthetises.reshape(-1, 3)[torch.randint(0, B * N, (N,))]
    #    .unsqueeze(0)
    #    .repeat(B, 1, 1)
    #)  # .repeat((B, N, 3))

    #can_base = can_base_hypthetises.mean(dim=0).unsqueeze(0).repeat(B, 1, 1)

    R_pred, t_pred, scale_pred, base_rot_residuum, canonical_base_center = (
        compute_base_pose(can_base, observed_base_pcl)
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

    #can_base = observed_base_pcl + deformation_field_base
    can_parts = deformation_fields_parts

    # Try fusing canonical across batch

    #can_base = can_base

    # can_parts = torch.cat(can_parts, dim=0)

    # can_base : (B*N, 3)
    # Choose randomly from combined cloud
    # can_base = can_base.reshape(-1, 3)[torch.randint(0, B * N, (N,))].unsqueeze(0).repeat(B, 1, 1)#.repeat((B, N, 3))

    # Combine all
    # can_base =  can_base.reshape(-1, 3).unsqueeze(0).expand(B, B*N ,3)

    if padding_size > 0:
        can_base = can_base[:, :-padding_size, :]
        can_parts = can_parts[:, :, :-padding_size, :]

    predicted_pcl, predicted_base_pcl = assemble_point_cloud(
        can_base,
        R_pred,
        scale_pred,
        t_pred,
        can_parts,
        thetas,
        pivot_points,
        rotation_axis,
        part_assingment,
    )

    return (
        predicted_pcl,
        base_rot_residuum,
        canonical_base_center,
        scale_pred,
        part_rot_residuum.sum(dim=1),
        part_non_axis_rot,
        can_base,
        can_parts,
        R_pred,
        scale_pred,
        t_pred,
        thetas,
        pivot_points,
        rotation_axis,
        predicted_base_pcl,
    )
