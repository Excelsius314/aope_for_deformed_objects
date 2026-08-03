import torch 

def farthest_point_sampling(D, K):
    """
    D: (B, N, 3)
    K: number of points to sample
    Returns:
        indices: (B, K)
    """
    B, N, _ = D.shape
    device = D.device

    # պահ sampled indices
    indices = torch.zeros(B, K, dtype=torch.long, device=device)

    # initialize distances to large values
    distances = torch.full((B, N), float('inf'), device=device)

    # randomly choose first point
    farthest = torch.randint(0, N, (B,), device=device)

    batch_indices = torch.arange(B, device=device)

    for i in range(K):
        indices[:, i] = farthest

        # get current farthest point coords
        centroid = D[batch_indices, farthest]  # (B, 3)

        # compute distances to all points
        dist = torch.sum((D - centroid.unsqueeze(1))**2, dim=-1)  # (B, N)

        # update minimum distances
        distances = torch.minimum(distances, dist)

        # pick next farthest point
        farthest = torch.argmax(distances, dim=1)

    return indices