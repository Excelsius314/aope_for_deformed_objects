import numpy as np

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
