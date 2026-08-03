import open3d as o3d
import os
from utility.transforms import homogenous_transform

def log_prediction(log_dir, epoch, target_pcl, predicted_pcl, predicted_canonical, comparable_canonical, last_cam_pose, last_predicted_base_pcl):
        pcl_log = o3d.geometry.PointCloud()

        epoch_log_dir = os.path.join(log_dir, "epoch_{}".format(epoch))
        os.makedirs(epoch_log_dir)

        pcl_log.points = o3d.utility.Vector3dVector(target_pcl)
        o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "observed_pcl_{}.ply".format(epoch)), pcl_log)

        pcl_log.points = o3d.utility.Vector3dVector(predicted_pcl)
        o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "predicted_pcl_{}.ply".format(epoch)), pcl_log)

        pcl_log.points = o3d.utility.Vector3dVector(predicted_canonical)
        o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "predicted_canonical_{}.ply".format(epoch)), pcl_log)

        pcl_log.points = o3d.utility.Vector3dVector(comparable_canonical)
        o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "comparable_canonical_cam_coords_{}.ply".format(epoch)), pcl_log)

        #comparable_canonical = homogenous_transform(last_cam_pose,comparable_canonical)
        #pcl_log.points = o3d.utility.Vector3dVector(comparable_canonical)
        #o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "comparable_canonical_world_coords_{}.ply".format(epoch)), pcl_log)

        pcl_log.points = o3d.utility.Vector3dVector(last_predicted_base_pcl[0].detach().cpu())
        o3d.io.write_point_cloud(os.path.join(epoch_log_dir, "predicted_non_normed_base{}.ply".format(epoch)), pcl_log)

def log_validation(log_dir, target_pcl, predicted_pcl):
        pcl_log = o3d.geometry.PointCloud()

        pcl_log.points = o3d.utility.Vector3dVector(target_pcl)
        o3d.io.write_point_cloud(os.path.join(log_dir, "observed_pcl.ply"), pcl_log)

        pcl_log.points = o3d.utility.Vector3dVector(predicted_pcl)
        o3d.io.write_point_cloud(os.path.join(log_dir, "predicted_pcl.ply"), pcl_log)