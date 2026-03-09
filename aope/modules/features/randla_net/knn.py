#
from typing import NamedTuple, Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.distances import BaseDistance, LpDistance

__all__ = ["KNN"]

def rm_kwargs(kwargs: Dict, keys: List):
    """Remove items corresponding to keys
    specified in 'keys' from kwargs dict."""
    keys_ = list(kwargs.keys())
    for k in keys:
        if k in keys_:
            del kwargs[k]
    return kwargs

class BaseDistance(torch.nn.Module):
    """

    Args:
        normalize_embeddings: flag to normalize provided embeddings
                                before calculating distances
        p: the exponent value in the norm formulation. (default: 2)
        power: If not 1, each element of the distance/similarity
                matrix will be raised to this power.
        is_inverted: Should be set by child classes.
                        If False, then small values represent
                        embeddings that are close together.
                        If True, then large values represent
                        embeddings that are similar to each other.
    """

    def __init__(
        self,
        normalize_embeddings: bool = True,
        p: Union[int, float] = 2,
        power: Union[int, float] = 1,
        is_inverted: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self._check_params()

    def _check_params(self):
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError(
                f"normalize_embeddings must be of type <bool>, "
                f"but got {type(self.normalize_embeddings)} instead."
            )
        if not (isinstance(self.p, (int, float))) or self.p <= 0:
            raise ValueError(f"p should be and int or float > 0, " f"but got {self.p}.")
        if not (isinstance(self.power, (int, float))) or self.power <= 0:
            raise ValueError(
                f"power should be and int or float > 0, " f"but got {self.power}."
            )
        if not isinstance(self.is_inverted, bool):
            raise ValueError(
                f"is_inverted must be of type <bool>, "
                f"but got {type(self.is_inverted)} instead."
            )

    def forward(self, query_emb: Tensor, ref_emb: Optional[Tensor] = None) -> Tensor:
        bs = query_emb.size(0)
        query_emb_normalized = self.maybe_normalize(query_emb, dim=-1)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb, dim=-1)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((bs, query_emb.size(1), ref_emb.size(1)))
        return mat

    def normalize(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def get_norm(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def compute_mat(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def maybe_normalize(self, embeddings: Tensor, dim: int = 1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=False, **kwargs)
        assert not self.is_inverted

    def compute_mat(
        self, query_emb: Tensor, ref_emb: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the batched p-norm distance between
        each pair of the two collections of row vectors."""
        if ref_emb is None:
            ref_emb = query_emb
        if query_emb.dtype == torch.float16:
            # cdist doesn't work for float16
            raise TypeError("LpDistance does not work for dtype=torch.float16")
        if len(query_emb.shape) == 2:
            query_emb = query_emb.unsqueeze(-1)
        if len(ref_emb.shape) == 2:
            ref_emb = ref_emb.unsqueeze(-1)
        assert len(query_emb.shape) == len(ref_emb.shape) == 3
        assert query_emb.size(-1) == ref_emb.size(-1) >= 1
        return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        """Computes the pairwise distance between
        vectors v1, v2 using the p-norm"""
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class KNeighbors(NamedTuple):
    """Named and typed result tuple for KNN search

    - distances: distance to each neighbor of each sample
    - indices: index of each neighbor of each sample
    - x_org: original x
    - x_norm: normalized x which was used for cluster centers and labels
    - k: number of neighbors

    """

    distances: Tensor
    indices: Tensor
    x_org: Tensor
    x_norm: Tensor
    k: Union[int, Tensor]


class KNN(nn.Module):
    """
    Implements k nearest neighbors in terms of
    pytorch tensor operations which can be run on GPU.
    Supports mini-batches of instances.

    Args:
        k: number of neighbors to consider
        distance: batched distance evaluator (default: LpDistance).
        p_norm: norm for lp distance (default: 2).
        normalize: String id of method to use to normalize input.
                        one of ['mean', 'minmax', 'unit'].
                        None to disable normalization. (default: None).

    """

    NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        k: int,
        distance: BaseDistance = LpDistance,
        p_norm: int = 2,
        normalize: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        super(KNN, self).__init__()
        self.k = k
        self.p_norm = p_norm
        self.normalize = normalize
        self._check_params()

        self.distance = distance(p=p_norm, **kwargs)
        self.eps = None

    def _check_params(self):
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k should be int > 0, but got {self.k}.")
        if self.p_norm <= 0:
            raise ValueError(f"p_norm should be > 0, but got {self.p_norm}.")
        if isinstance(self.normalize, bool):
            if self.normalize:
                self.normalize = "mean"
            else:
                self.normalize = None
        if self.normalize is not None and self.normalize not in self.NORM_METHODS:
            raise ValueError(
                f"unknown <normalize> method: {self.normalize}. "
                f"Please choose one of {self.NORM_METHODS}"
            )

    def _check_x(self, x) -> Tensor:
        """Check and (re-)format input samples x."""
        if not isinstance(x, Tensor):
            raise TypeError(f"x has to be a torch.Tensor but got {type(x)}.")
        shp = x.shape
        if len(shp) < 3:
            raise ValueError(
                f"input <x> should be at least of shape (BS, N, D) "
                f"with batch size BS, number of points N "
                f"and number of dimensions D but got {shp}."
            )
        elif len(shp) > 3:
            x = x.squeeze()
            x = self._check_x(x)
        self.eps = torch.finfo(x.dtype).eps
        return x

    def _check_k(self, k, dims: Optional[Tuple] = None) -> int:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k should be > 0, but got {k}.")
        if dims is not None:
            n = dims[1]
            if k >= n:
                raise ValueError(
                    f"k should be smaller than number " f"of samples {n} but got k={k}"
                )
        return k

    @torch.no_grad()
    def forward(
        self, x: Tensor, y: Tensor, k: Optional[int] = None, 
    ) -> KNeighbors:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional number of neighbors to use
            y: queries (BS, M, D)

        Returns:
            KNeighbors tuple

        """
        x = self._check_x(x)
        x_ = x
        k = self.k if k is None else k
        k = self._check_k(k, x.shape)
        # do not select self if from same source (instead of setting dist to inf)
        #same_source = int(same_source)
        #k += same_source
        # normalize input
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)

        values, indices = self.distance(x, y).sort(
            dim=-1, descending=self.distance.is_inverted
        )
        return KNeighbors(
            distances=values[:, :, : k],  # knn_distances
            indices=indices[:, :, : k],  # knn_indices
            x_org=x_,
            x_norm=x,
            k=k,
        )

    @staticmethod
    def _normalize(x: Tensor, normalize: str, eps: float = 1e-8):
        """Normalize input samples x according to specified method:

        - mean: subtract sample mean
        - minmax: min-max normalization subtracting sample min and divide by sample max
        - unit: normalize x to lie on D-dimensional unit sphere

        """
        if normalize == "mean":
            x -= x.mean(dim=1)[:, None, :]
        elif normalize == "minmax":
            x -= x.min(-1, keepdims=True).values  # type: ignore
            x /= x.max(-1, keepdims=True).values  # type: ignore
        elif normalize == "unit":
            # normalize x to unit sphere
            z_msk = x == 0
            x = x.clone()
            x[z_msk] = eps
            x = torch.diag_embed(1.0 / (torch.norm(x, p=2, dim=-1))) @ x
        else:
            raise ValueError(f"unknown normalization type {normalize}.")
        return x

    def fit(self, x: Tensor, k: Optional[int] = None, **kwargs) -> KNeighbors:
        """Compute k nearest neighbors for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional number of neighbors to use
            **kwargs: additional kwargs for fitting procedure

        Returns:
            KNeighbors tuple

        """
        return self(x, k=k, **kwargs)
