import torch
import torch.linalg
from torch.nn import MSELoss

from utility.transforms import homogenous_transform


def fully_alinged_point_could_loss(predicted_pcl, target_pcl):
    return torch.norm(predicted_pcl - target_pcl, dim=-1).sum(dim=1)

def pcl_similarity_loss_fn(predicted_pcl, target_pcl):
    #return chamfer_dist(predicted_pcl, target_pcl)
    return fully_alinged_point_could_loss(predicted_pcl, target_pcl)

def chamfer_dist(predicted_pcl, target_pcl):

    return fully_alinged_point_could_loss(predicted_pcl, target_pcl) 

    assert predicted_pcl.dim() == 3 and target_pcl.dim() == 3
    assert predicted_pcl.size(0) == target_pcl.size(0)
    assert predicted_pcl.size(2) == target_pcl.size(2)

    # Compute pairwise distances
    # (B, N, N)
    dist = torch.cdist(predicted_pcl, target_pcl, p=2)

    # For each point in x, find nearest neighbor in y
    min_x_to_y, _ = torch.min(dist, dim=2)  # (B, N)

    # For each point in y, find nearest neighbor in x
    min_y_to_x, _ = torch.min(dist, dim=1)  # (B, N)

    # Mean over points
    cd_x = min_x_to_y.mean(dim=1)  # (B,)
    cd_y = min_y_to_x.mean(dim=1)  # (B,)

    return cd_x + cd_y


def canonical_consistency_loss(predicted_pcl_batch, cam_poses):
    """
    Enforces canonical consistency using only unique pairs (upper triangle),
    reducing memory from O(B^2) to O(B*(B-1)/2).

    Args:
        predicted_pcl_batch: (B, N, 3) - batch of predicted point clouds
    Returns:
        Scalar loss - mean Chamfer distance over all unique pairs
    """
    B = predicted_pcl_batch.size(0)

    # Get upper triangle indices (excluding diagonal)
    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=predicted_pcl_batch.device)
    # idx_i, idx_j each have shape (B*(B-1)/2,)

    # Index directly — no expand/reshape over the full B*B grid
    pcl_i = predicted_pcl_batch[idx_i]  # (P, N, 3)
    pcl_j = predicted_pcl_batch[idx_j]  # (P, N, 3)

    # pcl_i = homogenous_transform(cam_poses[idx_i], predicted_pcl_batch[idx_i])
    # pcl_j = homogenous_transform(cam_poses[idx_j], predicted_pcl_batch[idx_j])

    # Reuse your existing loss, returns (P,)
    pair_losses =  chamfer_dist(pcl_i, pcl_j)

    return pair_losses.mean()

def pairwise_distance_consistency_loss_fn(observed_pcl, canonical_pcl, part_assignment, pred_scale, n_pairs=512):
    """
    Enforces that pairwise distances are preserved under the observed->canonical
    transformation, restricted to point pairs within the same rigid part.
    Valid for rigid transforms + uniform scale compensation per part.

    observed_pcl:    (B, N, 3) — points in observed space
    canonical_pcl:   (B, N, 3) — predicted points in canonical space
    part_assignment: (B, N)    — integer part id per point (e.g. 0=base, 1=handle)
    n_pairs:         number of point pairs to sample per batch element PER PART

    Returns a scalar loss.
    """
    B, N, _ = observed_pcl.shape

    device = observed_pcl.device

    part_ids = torch.unique(part_assignment)

    all_dist_observed = []
    all_dist_canonical = []
    valid_mask_list = []

    for part_id in part_ids:
        mask = (part_assignment == part_id)  # (B, N) bool

        # Number of points per sample belonging to this part
        counts = mask.sum(dim=1)  # (B,)
        max_count = counts.max().item()
        if max_count < 2:
            continue  # can't form a pair

        # Build per-sample index lists of points belonging to this part,
        # padded to max_count with index 0 (will be masked out later)
        part_indices = torch.zeros(B, max_count, dtype=torch.long, device=device)
        valid = torch.zeros(B, max_count, dtype=torch.bool, device=device)
        for b in range(B):
            idx_b = mask[b].nonzero(as_tuple=True)[0]  # indices of this part's points
            n_b = idx_b.shape[0]
            if n_b > 0:
                part_indices[b, :n_b] = idx_b
                valid[b, :n_b] = True

        # Sample pairs of LOCAL indices into part_indices (0..max_count-1)
        # then clamp to valid range per-sample using counts
        local_idx = torch.randint(0, max_count, (B, n_pairs, 2), device=device)
        # Clamp sampled local indices to each sample's actual count to avoid
        # picking padded/invalid slots; fall back to index 0 if count==1 (skipped above)
        counts_clamped = counts.clamp(min=1).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        local_idx = local_idx % counts_clamped

        local_a, local_b = local_idx[..., 0], local_idx[..., 1]  # (B, n_pairs)

        # Map local part-relative indices back to global point indices
        global_a = part_indices.gather(1, local_a)  # (B, n_pairs)
        global_b = part_indices.gather(1, local_b)

        # A pair is valid only if the two sampled local indices are different
        # (avoid zero self-distance pairs) and the sample actually has >=2 points
        pair_valid = (local_a != local_b) & (counts > 1).unsqueeze(-1)

        idx_a3 = global_a.unsqueeze(-1).expand(B, n_pairs, 3)
        idx_b3 = global_b.unsqueeze(-1).expand(B, n_pairs, 3)

        obs_a = observed_pcl.gather(1, idx_a3)
        obs_b = observed_pcl.gather(1, idx_b3)
        can_a = canonical_pcl.gather(1, idx_a3)
        can_b = canonical_pcl.gather(1, idx_b3)


        dist_observed = (obs_a - obs_b).norm(dim=-1) / pred_scale.squeeze(1)   # (B, n_pairs)
        dist_canonical = (can_a - can_b).norm(dim=-1)   # (B, n_pairs)

        all_dist_observed.append(dist_observed)
        all_dist_canonical.append(dist_canonical)
        valid_mask_list.append(pair_valid)

    if not all_dist_observed:
        return torch.tensor(0.0, device=device)
    
    print("All dist obs {}".format(all_dist_observed[0].shape))

    dist_observed = torch.cat(all_dist_observed, dim=1)
    dist_canonical = torch.cat(all_dist_canonical, dim=1)

    print("dist_observed shape: {}".format(dist_observed.shape))


    valid_mask = torch.cat(valid_mask_list, dim=1)

    diff = (dist_observed - dist_canonical).abs()
    print("Diff shape: {}".format(diff.shape))
    #diff = diff[valid_mask]

    print("Diff shape: {}".format(diff.shape))

    if diff.numel() == 0:
        return torch.tensor(0.0, device=device)

    return diff.sum(dim=1).mean()


def pairwise_distance_consistency_loss_fn2(observed_pcl, canonical_pcl, pred_scale):
    """
    Enforces that pairwise distances are preserved under the observed->canonical
    transformation (valid for rigid transforms + uniform scale compensation).

    observed_pcl:  (B, N, 3) — points in observed space
    canonical_pcl: (B, N, 3) — predicted points in canonical space
    n_pairs:       number of random point pairs to sample per batch element
                   (sampling avoids the O(N^2) cost of all pairs)

    Returns a scalar loss.
    """
    B, N, _ = observed_pcl.shape
    n_pairs = N//2

    # Sample random pairs of point indices
    idx = torch.randint(0, N, (B, n_pairs, 2), device=observed_pcl.device)
    idx_a, idx_b = idx[..., 0], idx[..., 1]  # (B, n_pairs) each

    # Gather the sampled points
    # expand idx to (B, n_pairs, 3) for gathering along dim=1
    idx_a3 = idx_a.unsqueeze(-1).expand(B, n_pairs, 3)
    idx_b3 = idx_b.unsqueeze(-1).expand(B, n_pairs, 3)

    obs_a  = observed_pcl.gather(1, idx_a3)   # (B, n_pairs, 3)
    obs_b  = observed_pcl.gather(1, idx_b3)
    can_a  = canonical_pcl.gather(1, idx_a3)
    can_b  = canonical_pcl.gather(1, idx_b3)

    # Pairwise distances in each space
    dist_observed  = (obs_a - obs_b).norm(dim=-1) / pred_scale   # (B, n_pairs)
    dist_canonical = (can_a - can_b).norm(dim=-1)   # (B, n_pairs)

    # If you have a per-sample scale factor s (B,) predicted separately,
    # the canonical distances should equal observed distances / s
    # Uncomment and pass scale if available:
    # dist_observed = dist_observed / scale.unsqueeze(-1)

    loss = (dist_observed - dist_canonical).abs().mean()
    return loss


def non_axis_alinged_rotation_loss(R_res):
    assert R_res.dim() == 4

    I = (
        torch.eye(3, device=R_res.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(R_res.shape[0], R_res.shape[1], 3, 3)
    )

    return torch.linalg.norm(R_res - I, dim=(-2, -1))


def canonical_zero_centered_loss(canonical_center, cam_poses):
    # return torch.linalg.norm(homogenous_transform(cam_poses, canonical_center), dim=-1)
    return torch.linalg.norm(canonical_center, dim=-1)


def normed_scale_loss_fn(predicted_canonical_scale):
    return torch.linalg.norm(1 - predicted_canonical_scale)


def pipeline_loss(
    predicted_pcl,
    base_rotation_residuum,
    part_rotation_residuums,
    part_non_axis_rot,
    target_pcl,
    center_canonical,
    scale_canonical,
    canonical_pcl,
    cam_poses,
    intermediate_feats,
    part_assingments,
    similarity_weight=2.0,
    base_rotation_weight=1.0,
    part_rotation_weight=1.0,
    canonical_zero_centered_weight=1.0,
    canonical_normed_scale_weight=1.0,
    canonical_consistency_weight=1.0,
):
    # Prediced pcl = Target pcl = Batch x N x 3
    assert predicted_pcl.dim() == 3
    #assert predicted_pcl.shape == target_pcl.shape
    assert predicted_pcl.shape[-1] == 3

    pcl_similarity_loss = pcl_similarity_loss_fn(predicted_pcl, target_pcl)
    # print("PCL similarity loss: ", pcl_similarity_loss)

    non_axis_rotation_loss = (
        non_axis_alinged_rotation_loss(part_non_axis_rot).sum(dim=1)
        + part_rotation_residuums
    )
    # print("Non-axis rotation loss: ", non_axis_rotation_loss.shape)
    # print("Part residuums: {}".format(part_rotation_residuums.shape))

    zero_centered_loss = canonical_zero_centered_loss(center_canonical, cam_poses)
    # print("Canonical zero-centered loss: ", zero_centered_loss)
    normed_scale_loss = normed_scale_loss_fn(scale_canonical)

    canonical_consistency = intermediate_feats#canonical_consistency_loss(intermediate_feats, cam_poses)

    point_distance_consistency_loss = pairwise_distance_consistency_loss_fn(target_pcl, canonical_pcl, part_assingments, scale_canonical)

    total_weight = (
        similarity_weight
        + base_rotation_weight
        + part_rotation_weight
        + canonical_zero_centered_weight
        + canonical_consistency_weight
        + canonical_normed_scale_weight
    )

    #weighted_loss = (
    #    similarity_weight / total_weight * pcl_similarity_loss
    #    + base_rotation_weight / total_weight * base_rotation_residuum
    #    + part_rotation_weight / total_weight * non_axis_rotation_loss
    #    + canonical_zero_centered_weight / total_weight * zero_centered_loss
    #    + canonical_normed_scale_weight / total_weight * normed_scale_loss
    #    + canonical_consistency_weight / total_weight * canonical_consistency
    #)

    weighted_loss = (
        similarity_weight / 1 * pcl_similarity_loss
        + base_rotation_weight / 1 * base_rotation_residuum
        + part_rotation_weight / 1 * non_axis_rotation_loss
        + canonical_zero_centered_weight / 1 * zero_centered_loss
        + canonical_normed_scale_weight / 1 * normed_scale_loss
        + canonical_consistency_weight / 1 * canonical_consistency
        + point_distance_consistency_loss * 0#10
    )

    #canonical_consistency = torch.tensor(0.0)
    return (
        weighted_loss.mean(),
        pcl_similarity_loss,
        base_rotation_residuum,
        non_axis_rotation_loss,
        zero_centered_loss,
        normed_scale_loss,
        canonical_consistency,
    )
