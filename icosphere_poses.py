import numpy as np
from icosphere import icosphere
from scipy.spatial.transform import Rotation as R

from pose_visualizer.pose_visualizer import PoseVisualizer
from pose_visualizer.camera_pose import CameraPose


def conversion_GL_CV(Tmxs: np.ndarray) -> np.ndarray:
    """Converts the camera pose from OpenGL to OpenCV notation

    Args:
        T (np.ndarray): Nx4x4 or 4x4 camera pose matrices

    Returns:
        np.ndarray: 4x4 camera pose matrix
    """
    Tmxs = Tmxs.reshape(-1, 4, 4)

    conversion_mtx = np.diag([1, -1, -1, 1]).reshape(1, 4, 4)
    Tmxs = Tmxs @ conversion_mtx
    
    if Tmxs.shape[0] == 1:
        Tmxs = Tmxs.squeeze()

    return Tmxs



def directed_angle(source_vector: np.ndarray, target_vector: np.ndarray, axis) -> float:
    # TODO: Vectorize this function so it works with sets of vectors
    """Calculates the angle from source_vector to target_vector around axis
    ONLY WORKS FOR VECTORS WHICH ARE ALREADY IN THE PLANE DEFINED BY AXIS

    Args:
        source_vector (np.ndarray): Vector to be aligned
        target_vector (np.ndarray): Target vector
        axis (np.ndarray): Axis to rotate around (normal to the plane), gives the orientation of the rotation

    Returns:
        float: np.ndarray
    """
    # Check if plane given by source and target is the same as the plane defined by axis
    # if np.dot(source_vector, axis) > 1e-10 or np.dot(target_vector, axis) > 1e-10:
    #     raise ValueError(
    #         "source and target vectors must be in the plane defined by axis"
    #     )

    # TODO: Add assertion for the input vectors

    source_vector = source_vector / np.linalg.norm(source_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    angle = np.arctan2(
        np.linalg.norm(np.cross(source_vector, target_vector)),
        np.dot(source_vector, target_vector),
    )

    # Check if the angle should be positive or negative
    triple_dot_product = np.dot(source_vector, np.cross(target_vector, axis))
    if triple_dot_product < 0:
        angle = -angle
    # print(angle)
    return angle


def generate_icosphere_views(
    nu: int = 2, center: np.ndarray = np.array([0, 0, 0]), radius: float = 1
) -> np.ndarray:
    # TODO: Try tp add paralization
    """Generates a set of camera poses around a sphere in icospere fashion
        The poses are generated in CV2 notation z is pointing towards the center of the sphere
        and x is parallel to the xy plane
        Number of verticies and faces in the icosphere is given by:
            nr_vertex = 12 + 10 * (nu**2 -1)
            nr_face = 20 * nu**2
        and is given by the icosphere function from https://pypi.org/project/icosphere/ module

    Args:
        nu (int, optional): Number of subdivisions. Defaults to 2.
        center (np.ndarray, optional): Center of the sphere. Defaults to np.array([0,0,0]).
        radius (float, optional): Radisu of the sphere. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    unit_vert, _ = icosphere(nu)  # 42 verticies

    # Preallocating the matricies
    T_mtxs = np.zeros((unit_vert.shape[0], 4, 4), dtype=np.float32)
    T_mtxs[:, :, :] = np.eye(4)
    T_mtxs[:, :3, 3] = unit_vert  # translation vectors

    # Generate poses so z is pointing towards the center of the sphere
    # and x is parallel to the xy plane
    z = np.array([0, 0, 1])
    y = np.array([0, 1, 0])
    z_dirs = -unit_vert
    y_dirs = z_dirs.copy()
    y_dirs[:, 2] = 0

    # NOTE: Might need to fasten this up by some vectorized operations
    for i in range(unit_vert.shape[0]):
        if np.allclose(y_dirs[i], np.array([0, 0, 0])):
            angle_z = 0
        else:
            angle_z = directed_angle(y, y_dirs[i], z)

        if angle_z is np.nan:
            angle_z = 0

        new_x = R.from_euler("Z", angle_z).apply(np.array([1, 0, 0]))

        angle_x = directed_angle(np.array([0, 0, 1]), z_dirs[i], new_x.flatten())

        # new_z = R.from_euler('ZYX', [angle_z, 0, angle_x]).apply(np.array([0,0,1]))

        R_mtx = R.from_euler("ZYX", [angle_z, 0, angle_x]).as_matrix()
        T_mtxs[i, :3, :3] = R_mtx

    # Debug visualization
    # pv = PoseVisualizer()
    # for i in range(unit_vert.shape[0]):
    #     cp = CameraPose()
    #     cp.Tmx = T_mtxs[i,:,:]
    #     cp.aspect_ratio = 1.0
    #     cp.units = "m"
    #     cp.visualization = "axes"
    #     pv.add_camera(cp)
    # pv.draw()

    T_mtxs[:, :3, 3] = T_mtxs[:, :3, 3] * radius + center

    return T_mtxs


if __name__ == "__main__":
    # vertices, faces = icosphere(2)
    # print(vertices, type(vertices))
    # print(faces, type(faces))
    # print(vertices.shape, faces.shape)
    # print(np.linalg.norm(vertices, axis=1))

    Tmxs = generate_icosphere_views(3)
    pv = PoseVisualizer()
    for i in range(Tmxs.shape[0]):
        cp = CameraPose()
        cp.Tmx = Tmxs[i,:,:]
        cp.aspect_ratio = 1.0
        cp.units = "m"
        cp.visualization = "axes"
        pv.add_camera(cp)

    Tmxs = conversion_GL_CV(Tmxs)
    for i in range(Tmxs.shape[0]):
        cp = CameraPose()
        cp.Tmx = Tmxs[i,:,:]
        cp.aspect_ratio = 1.0
        cp.units = "m"
        cp.visualization = "axes"
        pv.add_camera(cp)
    pv.draw()
