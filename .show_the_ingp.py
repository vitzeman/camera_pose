import json

import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def openGL2openCV(Tmx):
    """Convert OpenGL to OpenCV transformation matrix

    Args:
        Tmx (np.ndarray): Transformation matrix in OpenGL format

    Returns:
        np.ndarray: Transformation matrix in OpenCV format
    """
    Tmx_GL2CV = np.diag([1, -1, -1, 1])

    return Tmx @ Tmx_GL2CV


path2transforms = (
    "/home/vit/Mounts/ClusterHDDs/MLPrague25/Mustard/transform_finetuned.json"
)


with open(path2transforms, "r") as f:
    transforms = json.load(f)


to_draw = []
to_draw.append(
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
)  # Origin frame
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
sphere.paint_uniform_color([0, 0, 0])
to_draw.append(sphere)

frames = transforms["frames"]
for e, frame in enumerate(frames):

    Tmx = np.array(frame["transform_matrix"])

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    camera_frame.transform(Tmx)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.transform(Tmx)

    to_draw.append(camera_frame)
    to_draw.append(sphere)

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame.transform(openGL2openCV(Tmx))

    to_draw.append(camera_frame)

o3d.visualization.draw(to_draw)
