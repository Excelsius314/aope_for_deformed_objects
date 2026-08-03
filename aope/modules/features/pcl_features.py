from modules.features.randla_net.randla_model import RandLANet
from config.features import PCLFeatureConfig

import torch

class RANDLABackbone:
   
    def __init__(self, cfg : PCLFeatureConfig):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = RandLANet(
            d_in=cfg.backbone_config.d_in,
            num_neighbors=cfg.backbone_config.num_neighbors,
            decimation=cfg.backbone_config.decimation,
            feature_dim=cfg.backbone_config.feature_dim,
            device=torch.device(self.device)
        )

        self.model.eval().to(torch.device(self.device))

    def extract_features(self, point_cloud, part_embeddings=None):
        """
        Extract features from the input point cloud using the RandLA-Net backbone.

        Parameters
        ----------
        point_cloud: torch.Tensor, shape (B, N, d_in)
            Input point cloud data.

        Returns
        -------
        torch.Tensor, shape (B, N, feature_dim)
            Extracted features for each point in the point cloud.
        """
        features = self.model(point_cloud, part_embeddings).squeeze(-1).transpose(1, 2)
        return features
    
