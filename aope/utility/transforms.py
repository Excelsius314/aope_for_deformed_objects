import numpy as np
import torch

# Homogeneous transformation matrix
class T:

    def __init__(self):
        self.mat: np.ndarray = np.eye(4)

    def __matmul__(self, other):
        if isinstance(other, T):
            return T(self.mat @ other.mat)
        else:
            return self.mat @ other
    
    def __getitem__(self, val):
        return self.mat[val]
    
    def __setitem__(self, key, value):
        self.mat[key] = value

    def __str__(self):
        return str(self.mat)

    def __repr__(self):
        return repr(self.mat) + "\n"

    def transp(self):
        return T(self.mat.transpose())

    def inv(self):
        t_inv = T()
        t_inv[:3, :3] = self.mat[:3, :3].transpose()
        t_inv[:3, 3] = -t_inv[:3, :3] @ self.mat[:3, 3]
        return t_inv
    
    def jsonify(self):
        return self.mat[:3, :].tolist()

    @classmethod
    def from_rot_and_translation(
        cls, rot_mat: np.ndarray[(3, 3), float], translation_vec: np.ndarray
    ):
        t = T()
        t[:3, :3] = rot_mat
        t[:3, 3] = translation_vec
        return t

    # Create from matrix (R | t), add homogeneous row
    @classmethod
    def from_matrix(cls, mat: np.ndarray[(3, 4), float]):
        t = T()
        t.mat = np.eye(4)
        t.mat[:3, :4] = mat
        return t

    @classmethod
    def from_homog_transform(cls, homog_mat: np.ndarray[(4, 4), float]):
        t = T()
        t.mat = homog_mat
        return t

    @classmethod
    def from_euler_xyz(
        cls, euler_angles: np.ndarray[((3,))], translation_vec: np.ndarray[((3,))]
    ):
        t = T()

        rx, ry, rz = euler_angles
        t.mat[:3, :3] = np.array(
            [
                [np.cos(ry) * np.cos(rz), -np.cos(ry) * np.sin(rz), np.sin(ry)],
                [
                    np.sin(rx) * np.sin(ry) * np.cos(rz) + np.cos(rx) * np.sin(rz),
                    -np.sin(rx) * np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz),
                    -np.sin(rx) * np.cos(ry),
                ],
                [
                    -np.cos(rx) * np.sin(ry) * np.cos(rz) + np.sin(rx) * np.sin(rz),
                    np.cos(rx) * np.sin(ry) * np.sin(rz) + np.sin(rx) * np.cos(rz),
                    np.cos(rx) * np.cos(ry),
                ],
            ]
        )

        t.mat[:3, 3] = translation_vec

        return t
    
def homogenous_transform(t_mat : torch.Tensor, pcl : torch.Tensor):
    is_batched = len(pcl.shape) == 3
    if is_batched:
        # Input is batched
        ones_shape = (pcl.shape[0], pcl.shape[1], 1)
    else:
        ones_shape = (pcl.shape[0], 1)


    transformed_pcl = (t_mat @ torch.cat((pcl, torch.ones(ones_shape).to(pcl.device)), dim=-1).transpose(-1, -2)).transpose(-1, -2)

    if is_batched:
        return transformed_pcl[:, :, :-1]
    else:
        return transformed_pcl[:, :-1]
    
def axis_angle_to_rotation_matrix(a: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' formula, batched.

    Args:
        a:     (..., 3) rotation axes (need not be pre-normalized)
        theta: (...)    rotation angles in radians

    Returns:
        R: (..., 3, 3) rotation matrices
    """
    # Normalize the axis
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ax, ay, az = a.unbind(-1)
    zeros = torch.zeros_like(ax)

    # Skew-symmetric matrix K such that K @ v == a x v
    K = torch.stack([
        torch.stack([zeros, -az,  ay], dim=-1),
        torch.stack([az,  zeros, -ax], dim=-1),
        torch.stack([-ay,  ax, zeros], dim=-1),
    ], dim=-2)

    I = torch.eye(3, dtype=a.dtype, device=a.device).expand(*a.shape[:-1], 3, 3)

    sin_t = torch.sin(theta)[..., None, None]
    cos_t = torch.cos(theta)[..., None, None]

    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R

def pivot_transform(pivot : torch.Tensor, rot_axis: torch.Tensor, angle:torch.Tensor, pcl):

    R = axis_angle_to_rotation_matrix(rot_axis.squeeze(1), angle.squeeze(1).squeeze(1))

    print("Rotation mat shape {}".format(R.shape))
    return (R @ (pcl - pivot).transpose(-1, -2)).transpose(-1, -2) + pivot
