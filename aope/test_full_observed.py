import open3d as o3d
from open3d import io

from config.dataset import *
from data_processing.data_loading import ProcessedAOPEDataset
from torch.utils.data import DataLoader
from utility.transforms import homogenous_transform
import torch
import os

batch_size = 5

model = o3d.io.read_point_cloud("/home/marek/aope_for_deformed_objects/aope/data/valve_fixed.ply")
model = torch.tensor(model.points).unsqueeze(0)

model = model.expand((5,model.shape[1],3)).to("cuda:0").to(dtype=torch.float32)

print("Model shape:")
print(model.shape)

dataset_conf = DatasetConfig()
dataset_conf.path = (
    "/home/marek/aope_for_deformed_objects/aope/data/valve_fixed_preprocessed_5"
)
dataset_conf.pcl_size =1024


############## Load Dataset #####################
device = "cuda:0"

dataset = ProcessedAOPEDataset(dataset_conf, device)
loader = DataLoader(
    dataset,
    batch_size=5,
    shuffle=True,
    collate_fn=dataset.collate_fn,
)

test_dir = "/home/marek/aope_for_deformed_objects/aope/testing/test_data"

for batch in loader:

    obj_pose = batch["obj_poses"]
    observed = homogenous_transform(obj_pose, model)

    pcl_log = o3d.geometry.PointCloud()
    pcl_log.points = o3d.utility.Vector3dVector(observed[0].cpu().detach())
    o3d.io.write_point_cloud(os.path.join(test_dir, "full_observed_pcl.ply"), pcl_log)

    
    pcl_tensor = batch["pcls_cam_coords"]
    pcl_tensor = homogenous_transform(batch["cam_poses"], pcl_tensor)
    pcl_log.points = o3d.utility.Vector3dVector(pcl_tensor[0].cpu().detach())
    o3d.io.write_point_cloud(os.path.join(test_dir, "partial_from_data_pcl.ply"), pcl_log)

    break