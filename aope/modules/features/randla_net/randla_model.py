import time

import torch
import torch.nn as nn

# try:
#    from torch_points import knn
# except (ModuleNotFoundError, ImportError):
#    from torch_points_kernels import knn


#from pytorch3d.ops import knn_points as knn


import torch
from collections import namedtuple
from typing import Union

_KNN = namedtuple("KNN", "dists idx knn")


def knn(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,  # ignored, kept for API compatibility
    return_nn: bool = False,
    return_sorted: bool = True,
):
    """
    Pure PyTorch replacement for pytorch3d.ops.knn_points
    """

    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")
    if norm not in (1, 2):
        raise ValueError("Support for 1 (L1) or 2 (L2) norm only.")

    N, P1, D = p1.shape
    P2 = p2.shape[1]

    device = p1.device

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=device)

    # Compute pairwise distances
    if norm == 2:
        dists = torch.cdist(p1, p2, p=2) ** 2  # match squared L2
    else:
        dists = torch.cdist(p1, p2, p=1)

    # Mask invalid points in p2
    mask2 = torch.arange(P2, device=device)[None, :] >= lengths2[:, None]
    mask2 = mask2[:, None, :].expand(N, P1, P2)
    dists = dists.masked_fill(mask2, float("inf"))

    # Get K nearest
    knn_dists, knn_idx = torch.topk(dists, K, dim=2, largest=False, sorted=return_sorted)

    # Mask invalid points in p1
    mask1 = torch.arange(P1, device=device)[None, :] >= lengths1[:, None]
    mask1 = mask1[:, :, None].expand(N, P1, K)

    knn_dists = knn_dists.masked_fill(mask1, 0.0)
    knn_idx = knn_idx.masked_fill(mask1, 0)

    needs_mask = lengths2.min() < K
    if needs_mask:
        mask = lengths2[:, None] <= torch.arange(K, device=device)[None]
        mask = mask[:, None, :].expand(N, P1, K)
        knn_dists = knn_dists.masked_fill(mask, 0.0)
        knn_idx = knn_idx.masked_fill(mask, 0)

    # Optional neighbor gathering
    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, knn_idx, lengths2)

    return _KNN(dists=knn_dists, idx=knn_idx, knn=p2_nn)

def knn_gather(
    x: torch.Tensor,
    idx: torch.Tensor,
    lengths: Union[torch.Tensor, None] = None,
):
    N, M, U = x.shape
    _, L, K = idx.shape

    if lengths is None:
        lengths = torch.full((N,), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    needs_mask = lengths.min() < K
    if needs_mask:
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode="zeros",
        bn=False,
        activation_fn=None,
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode,
        )
        self.batch_norm = (
            nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        )
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
        Forward pass of the network

        Parameters
        ----------
        input: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
        Forward pass

        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d, N, 1)
            features of the point cloud
        neighbors: tuple

        Returns
        -------
        torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx = knn_output.idx
        dist = knn_output.dists
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat(
            (
                extended_coords,
                neighbors,
                extended_coords - neighbors,
                dist.unsqueeze(-3),
            ),
            dim=-3,
        ).to(self.device)
        return torch.cat((self.mlp(concat), features.expand(B, -1, N, K)), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(
            in_channels, out_channels, bn=True, activation_fn=nn.ReLU()
        )

    def forward(self, x):
        r"""
        Forward pass

        Parameters
        ----------
        x: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


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


class SmallRandla(nn.Module):
    def __init__(
        self, d_in, num_neighbors=16, decimation=4, feature_dim=32, device=torch.device("cpu")
    ):
        super(SmallRandla, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 64)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-6, momentum=0.99), nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList(
            [
                LocalFeatureAggregation(64, 128, num_neighbors, device),   # in 64  -> out 256 (2*128)
                LocalFeatureAggregation(256, 128, num_neighbors, device),  # in 256 -> out 256
                LocalFeatureAggregation(256, 256, num_neighbors, device),
                LocalFeatureAggregation(512, 512, num_neighbors, device)
            ]
        )

        self.mlp = SharedMLP(1024, 1024, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.decoder = nn.ModuleList(
            [
                SharedMLP(1024, 512, **decoder_kwargs),
                SharedMLP(512, 256, **decoder_kwargs),
                SharedMLP(256, 128, **decoder_kwargs),
                SharedMLP(128, feature_dim, **decoder_kwargs),
            ]
        )

        self.device = device

        self = self.to(device)

    def forward(self, input):
        r"""
        Forward pass

        Parameters
        ----------
        input: torch.Tensor, shape (B, N, d_in)
            input points

        Returns
        -------
        torch.Tensor, shape (B, N)
            segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone()#.cpu()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            x = lfa(coords[:, : N // decimation_ratio], x)
            #x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        bottle_neck = self.mlp(x)
        x = bottle_neck

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:

            knn_output = knn(
                coords[:, : d * N // decimation_ratio].contiguous(),
                coords[:, : N // decimation_ratio].contiguous(),
                K=1,
            )

            neighbors = knn_output.idx  # (B, N, 1)

            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            #x_neighbors = torch.gather(x, -2, extended_neighbors)
            x = torch.gather(x, -2, extended_neighbors)

            #x = torch.cat((x_neighbors, x_stack.pop()), dim=1)
            
            x = mlp(x)

            decimation_ratio //= d

        return x, bottle_neck


class RandLANet(nn.Module):
    def __init__(
        self, d_in, num_neighbors=16, decimation=4, feature_dim=32, device=torch.device("cpu")
    ):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            #nn.BatchNorm2d(8, eps=1e-6, momentum=0.99), nn.LeakyReLU(0.2)
            #nn.BatchNorm2d(8, eps=1e-6, momentum=0.1),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList(
            [
                LocalFeatureAggregation(16, 16, num_neighbors, device),
                LocalFeatureAggregation(32, 64, num_neighbors, device),
                LocalFeatureAggregation(128, 128, num_neighbors, device),
                LocalFeatureAggregation(256, 256, 8, device),
            ]
        )

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.decoder = nn.ModuleList(
            [
                SharedMLP(1024, 256, **decoder_kwargs),
                SharedMLP(512, 128, **decoder_kwargs),
                SharedMLP(256, 32, **decoder_kwargs),
                SharedMLP(64, feature_dim, **decoder_kwargs),
            ]
        )

        # final semantic prediction
        # self.fc_end = nn.Sequential(
        #    SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
        #    SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
        #    nn.Dropout(),
        #    SharedMLP(32, num_classes)
        # )
        self.device = device

        self = self.to(device)

    def forward(self, input, part_embeddings=None):
        r"""
        Forward pass

        Parameters
        ----------
        input: torch.Tensor, shape (B, N, d_in)
            input points

        Returns
        -------
        torch.Tensor, shape (B, N)
            segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone()#.cpu()
        x = (torch.cat((self.fc_start(input), part_embeddings), dim=-1)).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            print("Encode: {}".format(x.shape))
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, : N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        print("Bottleneck {}".format(x.shape))

        x = self.mlp(x)

        print("Mlp bottleneck {}".format(x.shape))

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            #neighbors, _ = knn(
            #    coords[:, : N // decimation_ratio].cpu().contiguous(),  # original set
            #    coords[:, : d * N // decimation_ratio]
            #    .cpu()
            #    .contiguous(),  # upsampled set
            #    1,
            #)  # shape (B, N, 1)

            knn_output = knn(
                coords[:, : d * N // decimation_ratio].contiguous(),
                coords[:, : N // decimation_ratio].contiguous(),
                K=1,
            )

            neighbors = knn_output.idx  # (B, N, 1)

            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        # x = x[:,:,torch.argsort(permutation)]

        # scores = self.fc_end(x)

        # return scores.squeeze(-1)

        return x


if __name__ == "__main__":
    import time

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d_in = 3
    cloud = 1000 * torch.randn(1, 2**16, d_in).to(device)
    model = RandLANet(d_in, 16, 4, feature_dim=32, device=device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    # print(pred)
    #print(t1 - t0)
    #print(pred.shape)
    #print(pred[0, :, 0, 0])
