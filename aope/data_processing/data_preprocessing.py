import numpy as np
import os
import cv2
from typing import List
import open3d as o3d
import shutil
import torch
import zarr

from tqdm import tqdm

from utility.transforms import T
from utility.json import write_to_json

from config.segmentation import SegmentationConfig
from config.pipeline import PreprocessingConfig
from data_processing.data_loading import AOPEDataset
from modules.segmentation.segmentation import segment




from config.features import VisualFeatureConfig
from modules.features.visual_features import VisualFeaturePreprocessor


class AOPEPreprocesser:

    def __init__(
        self,
        dataset: AOPEDataset,
        preprocessing_config: PreprocessingConfig,
        v_feat_conf: VisualFeatureConfig = None,
    ):
        self.config = preprocessing_config
        self.dataset = dataset

        self.pcls_cam_coords = []
        self.pcls_pixel_coords = []
        self.part_assingments = []
        self.cam_poses = []
        self.obj_poses = []
        self.img_features = None
        self.segmentations = []
        self.processed_labels = self.dataset.labels
        self.joint_states = []

        if self.config.preprocess_visual_features:
            if not v_feat_conf:
                v_feat_conf = VisualFeatureConfig()
            self.vis_preprocessor = VisualFeaturePreprocessor(v_feat_conf)

    def project_point_clouds_to_cam_coords(
        self, pcl: np.ndarray, cam_pose: np.array, idPreprocessingConfigx: int
    ) -> List[np.ndarray]:

        if self.config.flip_cam_x_axis:
            cam_pose[:, 0] *= -1  # Invert x axis

        # Convert points to homogeneous coordinates
        points_homogeneous = np.hstack((pcl, np.ones((pcl.shape[0], 1))))
        # Apply the inverse camera pose transformation
        return (cam_pose.inv() @ points_homogeneous.T).T[:, :3]

    def project_point_clouds_to_img_coords(
        self, pcl: np.ndarray, cam_intrinsics: np.ndarray, idx: int
    ):
        transformed_points = (cam_intrinsics @ pcl.T).T
        pixel_coords = transformed_points[:, :2] / transformed_points[:, 2:]

        if self.config.create_debug_imgs:
            self.create_projection_debug_img(
                idx, self.dataset.config.imgsz, pixel_coords
            )

        return pixel_coords

    def assing_parts_to_points(self, pcl_pixel_coords, segmentation, idx):
        part_assignments = np.full((pcl_pixel_coords.shape[0], 1), -1, dtype=int)

        # Segment with model
        if self.config.preprocess_segmentations:
            for cls_idx, cls_seg in zip(
                segmentation[0].boxes.cls, segmentation[0].masks.data.cpu()
            ):
                indices = np.round(pcl_pixel_coords).astype(int)
                part_assignments[
                    np.nonzero(cls_seg.numpy()[indices[:, 1], indices[:, 0]])
                ] = int(cls_idx)

            if self.config.create_debug_imgs:
                self.create_segmentation_debug_img(
                    idx, self.dataset.config.imgsz, pcl_pixel_coords, part_assignments
                )
        else:
            # Use ground truth seg maps
            indices = np.round(pcl_pixel_coords).astype(int)
            part_assignments = np.load(segmentation)[indices[:, 1], indices[:, 0]]

            if self.config.create_debug_imgs:
                self.create_segmentation_debug_img(
                    idx, self.dataset.config.imgsz, pcl_pixel_coords, part_assignments
                )

        return part_assignments

    def run(self):

        batch_size = self.config.batch_size

        if self.config.save_back_to_disk:
            dataset_dir, img_dir, pcl_dir = self.create_save_dir()

        if self.config.preprocess_segmentations:
            self.segmentations = segment(
                self.dataset.img_files, self.config.segmentation_config
            )
        else:
            seg_map_dir = os.path.join(self.dataset.config.path, "seg_maps")
            self.segmentations = [
                os.path.join(seg_map_dir, file)
                for file in sorted(os.listdir(seg_map_dir))
            ]

        for k in tqdm(range(0, min(len(self.dataset), self.config.max_samples), batch_size)):
            for idx in range(k, k + batch_size):
                data_sample = self.dataset[idx]

                self.pcls_cam_coords.append(
                    self.project_point_clouds_to_cam_coords(
                        np.asarray(data_sample["pcl"].points),
                        data_sample["cam_poses"],
                        idx,
                    )
                )

                self.pcls_pixel_coords.append(
                    self.project_point_clouds_to_img_coords(
                        self.pcls_cam_coords[-1], data_sample["cam_intrinsics"], idx
                    )
                )

                self.part_assingments.append(
                    self.assing_parts_to_points(
                        self.pcls_pixel_coords[-1], self.segmentations[idx], idx
                    )
                )

                self.obj_poses.append(data_sample["obj_pose_mat"].mat)
                cam_pose = data_sample["cam_poses"]
                if self.config.flip_cam_x_axis:
                    cam_pose[:, 0] *= -1  # Invert x axis

                self.joint_states.append(data_sample["joint_states"])

                self.cam_poses.append(cam_pose.mat)

            if self.config.save_back_to_disk:
                if self.config.preprocess_visual_features:
                    imgs = [
                        torch.Tensor(cv2.imread(self.dataset.load_img(idx))).unsqueeze(
                            0
                        )
                        for idx in range(k, k + batch_size)
                    ]
                    imgs = torch.cat(imgs, dim=0)
                    self.img_features = self.vis_preprocessor.model.get_local_features(
                        imgs
                    )
                

                self.write_data_to_save_dir(dataset_dir, img_dir, pcl_dir, k)

                self.pcls_cam_coords.clear()
                self.pcls_pixel_coords.clear()
                self.part_assingments.clear()
                self.cam_poses.clear()
                self.obj_poses.clear()

                print(self.joint_states)
                self.joint_states.clear()

    def create_projection_debug_img(self, idx, imgsize, coords):
        if isinstance(imgsize, tuple):
            pseudo_depth_image = np.zeros(imgsize, dtype=np.uint8)
        elif isinstance(imgsize, int):
            pseudo_depth_image = np.zeros((imgsize, imgsize), dtype=np.uint8)
        else:
            print("Invalid image size: {}".format(imgsize))
            return

        for transformed_point in coords:
            x, y = int(transformed_point[0]), int(transformed_point[1])
            if (
                0 <= x < pseudo_depth_image.shape[1]
                and 0 <= y < pseudo_depth_image.shape[0]
            ):
                pseudo_depth_image[y, x] = 255
            else:
                print("Warning: pixel coordinate out of bounds:", transformed_point)
        success = cv2.imwrite(
            os.path.join(
                self.config.debug_path, "projected_points_debug_{}.png"
            ).format(idx),
            pseudo_depth_image,

            print(os.path.join(
                self.config.debug_path, "projected_points_debug_{}.png"
            ).format(idx))
        )

        if not success:
            print("Warning: Failed to writed debug img")

    def create_segmentation_debug_img(self, idx, imgsize, coords, segmentation):
        if isinstance(imgsize, tuple):
            pcl_seg_img = np.full(imgsize, 255, dtype=int)
        elif isinstance(imgsize, int):
            pcl_seg_img = np.full((imgsize, imgsize, 3), 255, dtype=int)
        else:
            print("Invalid image size: {}".format(imgsize))
            return
        

        for i, pixel_coord in enumerate(coords):
            x, y = int(np.round(pixel_coord[0])), int(np.round(pixel_coord[1]))
            if 0 <= x < pcl_seg_img.shape[1] and 0 <= y < pcl_seg_img.shape[0]:

                seg_val = (segmentation[i])
                color_val = [0, 0, 0]
                if seg_val == 1:
                    color_val = [255, 0, 0]
                else:
                    color_val = [0, 0, 255]

                pcl_seg_img[y, x] = color_val 
            else:
                print("Warning: pixel coordinate out of bounds:", pixel_coord)


        # Normalize
        #pcl_seg_img = ((pcl_seg_img + 1) / np.max(pcl_seg_img + 1)) * 255

        cv2.imwrite(
            os.path.join(self.config.debug_path, "segmented_point_cloud_{}.png").format(
                idx
            ),
            pcl_seg_img.astype(np.uint8)#cv2.applyColorMap(pcl_seg_img.astype(np.uint8), cv2.COLORMAP_JET),
        )

    def create_save_dir(self):
        dataset_dir = os.path.basename(self.dataset.config.path) + "_preprocessed"

        # Check if previous runs of preprocessed were saved
        idx = len(
            [
                dir_name
                for dir_name in os.listdir(self.config.save_dir)
                if dir_name.startswith(dataset_dir)
            ]
        )
        if idx > 0:
            dataset_dir = dataset_dir + "_{}".format(idx + 1)

        dataset_dir = os.path.join(self.config.save_dir, dataset_dir)
        pcl_dir = os.path.join(dataset_dir, "point_clouds")
        img_dir = os.path.join(dataset_dir, "images")
        # label_file = os.path.join(dataset_dir, "scene_data.json")

        # Create dataset infrastructure
        os.makedirs(dataset_dir)
        os.makedirs(img_dir)
        os.makedirs(pcl_dir)

        return dataset_dir, img_dir, pcl_dir

    def write_data_to_save_dir(self, dataset_dir, img_dir, pcl_dir, idx_offset):

        # Write back pcls in cam coordinates
        for idx, pcl in enumerate(self.pcls_cam_coords):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)

            file_name = "{:06d}.ply".format(idx + idx_offset)
            o3d.io.write_point_cloud(os.path.join(pcl_dir, file_name), pcd)

        # Just copy img files for
        for img_file in self.dataset.img_files:
            shutil.copy(img_file, os.path.join(img_dir, os.path.basename(img_file)))

        if self.config.preprocess_segmentations:
            seg_dir = os.path.join(dataset_dir, "segmentations")
            os.makedirs(seg_dir)

            # Save segmentation data
            for idx, segmentation in enumerate(self.segmentations):
                seg_file = os.path.join(seg_dir, "{:06d}.json".format(idx))
                open(seg_file, "a").close()

                write_to_json(
                    {
                        "classes": segmentation[0].boxes.cls.cpu().numpy().astype(int),
                        "seg_masks": segmentation[0]
                        .masks.data.cpu()
                        .numpy()
                        .astype(int),
                    },
                    seg_file,
                )
        processed_data = {
            "pcls_cam_coords": self.pcls_cam_coords,
            "pcls_pixel_coords": self.pcls_pixel_coords,
            "part_assingments": self.part_assingments,
            "cam_poses" : self.cam_poses,
            "obj_poses" : self.obj_poses,
            "joint_states" : self.joint_states
        }
        if self.config.preprocess_visual_features:
            processed_data["img_features"] = self.img_features.numpy()


        self.write_to_zarr(
            processed_data,
            dataset_dir,
            True,
        )

    def write_to_zarr(
        self, data, data_path, append_to_existing=False, root_group: zarr.Group = None
    ):
        if root_group is not None:
            z_group = root_group
        else:
            z_group: zarr.Group = zarr.open_group(data_path, mode="a")

        for label in data:
            if type(data[label]) is dict:

                if append_to_existing and label in z_group:
                    sub_group = z_group[label]
                else:
                    sub_group = z_group.create_group(label, overwrite=True)

                self.write_to_zarr(
                    data[label], data_path, append_to_existing, sub_group
                )
            else:

                arr = np.array(data[label])
                if label in z_group:
                    if append_to_existing:
                        z_group[label].append(data[label], axis=0)
                    else:
                        z_group.create_dataset(
                            name=label, data=arr, shape=arr.shape, overwrite=True
                        )
                else:
                    z_group.create_dataset(name=label, data=arr, shape=arr.shape)

        return z_group


def normalize_point_clouds(point_clouds: np.ndarray, cam_poses) -> np.ndarray:
    # points_clouds: (B, N, 3)
    centers = np.mean(point_clouds, axis=1, keepdims=True)
    assert centers.shape == (point_clouds.shape[0], 1, 3)
    return point_clouds - centers
