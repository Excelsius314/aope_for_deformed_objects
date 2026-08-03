import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch


from modules.features.randla_net.randla_model import (
    LocalSpatialEncoding,
    AttentivePooling,
    knn,
    SharedMLP,
)
from utility.math import farthest_point_sampling


def apply_part_mlp(point_features, part_assignments, mlp, K):
    B, N, C = point_features.shape

    part_ids = part_assignments.squeeze(-1)
    mask = torch.nn.functional.one_hot(part_ids, K).permute(0, 2, 1).float()

    x = point_features.unsqueeze(1) * mask.unsqueeze(-1)  # (B, K, N, C)

    x = x.view(B, K, N * C)
    out = mlp(x)
    out = out.view(B, K, N, -1)

    # enforce masking
    out = out * mask.unsqueeze(-1)

    return out


def apply_part_mlp_per_point(point_features, part_assignments, mlp, K):
    B, N, C = point_features.shape

    part_ids = part_assignments.squeeze(-1)

    mask = torch.nn.functional.one_hot(part_ids, K).permute(0, 2, 1).float()

    x = point_features.unsqueeze(1) * mask.unsqueeze(-1)  # (B, K, N, C)

    out = mlp(x)
    out = out.view(B, K, N, -1)

    # enforce masking
    out = out * mask.unsqueeze(-1)

    return out


# Computes per-point transformation from per-point feature
class ShapeReconstruction(nn.Module):

    def __init__(self, n_points, feature_dim, K):
        super(ShapeReconstruction, self).__init__()

        self.feature_dim = feature_dim
        self.n_points = n_points
        self.K = K

        # Regress per Point
        self.fn1 = nn.Linear(feature_dim, 32)
        self.fn2 = nn.Linear(32, 16)
        self.fn_out = nn.Linear(16, 3)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 16),
                nn.Linear(16, 3)
            )
            for _ in range(K)
        ])

    def forward(self, x, part_ids):
        """
        x:        (B, N, feature_dim)
        part_ids: (B, N)  — values in [0, K)
        returns:  (B, N, out_dim)
        """
        B, N, _ = x.shape
        out_dim = self.mlps[0][-1].out_features
        #out = torch.zeros(B, N, out_dim, device=x.device, dtype=x.dtype)

        results = [torch.zeros(B, N, out_dim, device=x.device, dtype=x.dtype) for i in range(self.K)]

        for k, mlp in enumerate(self.mlps):
            mask = (part_ids == k)          # (B, N) bool
            if not mask.any():
                continue

            pts = x[mask]                   # (M_k, feature_dim) — gathered points
            result = mlp(pts)               # (M_k, out_dim)

            results[k][mask] = result
            #out[mask] = result              # scatter back

        return results


class LocalPointAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, reduction_ratio, device):
        super(LocalPointAggregation, self).__init__()

        self.num_neighbors = num_neighbors
        self.reduction_ratio = reduction_ratio

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device).to(device)

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2)).to(
            device
        )
        self.mlp2 = SharedMLP(d_out, d_out, activation_fn=nn.LeakyReLU(0.2)).to(device)

        self.pool = AttentivePooling(d_out, d_out).to(device)

    def forward(self, features, coords):
        r"""
        Forward pass

        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d_in, N, 1)
            features of the point cloud

        Returns
        -------
        torch.Tensor, shape (B, 2*d_out, N, 1)
        """

        B, N_sub, K = (
            features.shape[0],
            features.shape[2] // self.reduction_ratio,
            self.num_neighbors,
        )

        knn_output = knn(
            coords.contiguous(),
            coords.contiguous(),
            K=self.num_neighbors,
        )

        x = self.mlp1(features)  # (B, N, d_out//2)
        x = self.lse1(coords, x, knn_output)  # (B, d_out, N, K)

        # PointNet style subsampling
        subsample_indices = farthest_point_sampling(coords, N_sub)  # (B, N_sub)
        x = torch.gather(
            x,
            2,
            subsample_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, x.shape[1], N_sub, K),
        )  # (B, d_in,  N //self.reduction_ratio, K)
        coords = torch.gather(
            coords, 1, subsample_indices.unsqueeze(-1).expand(B, N_sub, 3)
        )
        x = self.pool(x)  # (B, d_out, N, 1)

        return self.mlp2(x), coords


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
        Forward pass

        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d_in, N, 1)
            features of the point cloud

        Returns
        -------
        torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(
            coords.contiguous(),
            coords.contiguous(),
            K=self.num_neighbors,
        )

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)

        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class JointParameterPediction(nn.Module):

    def __init__(
        self, d_in, num_neighbors, reduction_ratio=8, num_reductions=3, device="cpu"
    ):
        super(JointParameterPediction, self).__init__()

        self.reductions = [
            LocalPointAggregation(d_in, 64, num_neighbors, reduction_ratio, device),
            LocalPointAggregation(64, 128, num_neighbors, reduction_ratio, device),
            LocalPointAggregation(128, 256, num_neighbors, reduction_ratio, device),
            LocalPointAggregation(256, 512, reduction_ratio, reduction_ratio, device),
        ]

        self.mlp = nn.Linear(512, 256).to(device)
        self.ac_fn = nn.LeakyReLU(0.2)

        # Pivot point head
        self.pv_head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.Linear(128, 32, bias=False),
            nn.Linear(32, 3, bias=False),
        ).to(device)

        # Rotation axis head
        self.axis_head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.Linear(128, 32, bias=False),
            nn.Linear(32, 3, bias=False),
        ).to(device)

    def forward(self, x, coords):
        """
        x : (B, N, D)
        """

        x = torch.permute(x.unsqueeze(-1), (0, 2, 1, 3))  # (B, D, N, 1)
        # Regress from two adjacent parts
        for reduction_layer in self.reductions:
            x, coords = reduction_layer(x, coords)

        x = x.squeeze(-1).squeeze(-1)

        x = self.ac_fn(self.mlp(x.sum(dim=-1)))

        return torch.cat((self.pv_head(x), self.axis_head(x)), dim=1)
