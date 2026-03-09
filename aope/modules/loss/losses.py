import torch
import torch.linalg


def point_cloud_loss(predicted_pcl, target_pcl):

    assert predicted_pcl.dim() == 3 and target_pcl.dim() == 3
    assert predicted_pcl.size(0) == target_pcl.size(0)
    assert predicted_pcl.size(2) == target_pcl.size(2)

    # Compute pairwise distances
    # (B, N, N)
    dist = torch.cdist(predicted_pcl, target_pcl, p=2) ** 2

    # For each point in x, find nearest neighbor in y
    min_x_to_y, _ = torch.min(dist, dim=2)  # (B, N)

    # For each point in y, find nearest neighbor in x
    min_y_to_x, _ = torch.min(dist, dim=1)  # (B, N)

    # Mean over points
    cd_x = min_x_to_y.mean(dim=1)  # (B,)
    cd_y = min_y_to_x.mean(dim=1)  # (B,)

    return (cd_x + cd_y).mean()  # Scalar loss mean over batch


def non_axis_alinged_rotation_loss(R_res):
    assert R_res.dim() == 4

    I = (
        torch.eye(3, device=R_res.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(R_res.shape[0], R_res.shape[1], 3, 3)
    )

    return torch.linalg.norm(R_res - I, dim=(-2, -1))


def canonical_zero_centered_loss(canonical_center):
    return torch.linalg.norm(canonical_center, dim=-1).mean()


def pipeline_loss(
    predicted_pcl,
    base_rotation_residuum,
    part_rotation_residuums,
    target_pcl,
    center_canonical,
    similarity_weight=1.0,
    base_rotation_weight=1.0,
    part_rotation_weight=1.0,
    canonical_zero_centered_weight=1.0,
):
    # Prediced pcl = Target pcl = Batch x N x 3
    assert predicted_pcl.dim() == 3
    assert predicted_pcl.shape == target_pcl.shape
    assert predicted_pcl.shape[-1] == 3

    pcl_similarity_loss = point_cloud_loss(predicted_pcl, target_pcl)
    print("PCL similarity loss: ", pcl_similarity_loss)

    non_axis_rotation_loss = non_axis_alinged_rotation_loss(
        part_rotation_residuums
    ).sum(dim=1)
    print("Non-axis rotation loss: ", non_axis_rotation_loss)

    zero_centered_loss = canonical_zero_centered_loss(center_canonical)
    print("Canonical zero-centered loss: ", zero_centered_loss)

    total_weight = (
        similarity_weight
        + base_rotation_weight
        + part_rotation_weight
        + canonical_zero_centered_weight
    )

    return (
        similarity_weight / total_weight * pcl_similarity_loss
        + base_rotation_weight / total_weight * base_rotation_residuum
        + part_rotation_weight / total_weight * non_axis_rotation_loss
        + canonical_zero_centered_weight / total_weight * zero_centered_loss
    ).mean()
