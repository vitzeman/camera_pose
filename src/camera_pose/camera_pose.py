from dataclasses import dataclass, field
from typing import List, Any
import json


import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

M2MM = 1000.0
MM2M = 0.001


@dataclass
class CameraPose:
    """
    Data class to store the camera pose
    # TODO: Add all the atributes to the docstring
    Attributes:
    Tmx: np.ndarray: 4x4 camera pose w.r.t. the world frame, Default: Identity matrix (4x4)
    resolution: tuple: resolution of the camera image
    aspect_ratio: float: aspect ratio of the camera image
    units: str: units of the translation (mm or m) # TODO: Not properly implemented
    visualization: str: visualization type (both, pyramid, axes)
    camera_notation: str: camera notation (opencv, opengl)
    image_path: str: path to the camera image
    depth_path: str: path to the depth image
    mask_path: str: path to the mask image
    T_d2c: np.ndarray: 4x4 transformation matrix from depth to color camera frame
    name: str: camera name
    color: list: color of the camera model (RGB), Default: [0, 0, 0] (black)
    """

    Tmx: np.ndarray = field(default_factory=lambda: np.eye(4))
    aspect_ratio: float = 1.0
    resolution: tuple = None
    color_K: np.ndarray = None
    color_dist: np.ndarray = None
    depth_K: np.ndarray = None
    depth_dist: np.ndarray = None
    
    units: str = "m"
    visualization: str = "both"
    camera_notation: str = "opencv"
    image_path: str = None
    depth_path: str = None
    depth_scale: float = 1.0
    mask_path: str = None
    T_d2c: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = None
    color: list = field(default_factory=lambda: [0, 0, 0])

    def __setattr__(self, name, value) -> None:
        if name == "Tmx":
            assert type(value) == np.ndarray
            if value.shape != (4, 4):
                raise ValueError("Tmx must be a 4x4 matrix")
        elif name == "aspect_ratio":
            if value <= 0:
                raise ValueError("Aspect ratio must be positive")
        elif name == "resolution":
            if type(value) not in [tuple, type(None)]:
                raise ValueError("Resolution must be a tuple")
            if value is not None:
                if len(value) != 2:
                    raise ValueError("Resolution must be a tuple of 2 elements")
                if not all([type(val) == int for val in value]):
                    raise ValueError("Resolution values must be integers")
                if not all([val > 0 for val in value]):
                    raise ValueError("Resolution values must be positive")
        elif name == "color_K":
            if type(value) not in [np.ndarray, type(None)]:
                raise ValueError("Intrinsic matrix must be a numpy array")
            if value is not None:
                if value.shape != (3, 3):
                    raise ValueError("Intrinsic matrix must be a 3x3 matrix")
        elif name == "color_dist":
            if type(value) not in [np.ndarray, type(None)]:
                raise ValueError("Distortion coefficients must be a numpy array")
            if value is not None:
                if value.shape[0] != 5:
                    raise ValueError("Distortion coefficients must be a 5 element array")
                
        elif name == "depth_K":
            if type(value) not in [np.ndarray, type(None)]:
                raise ValueError("Intrinsic matrix must be a numpy array")
            if value is not None:
                if value.shape != (3, 3):
                    raise ValueError("Intrinsic matrix must be a 3x3 matrix")
        elif name == "depth_dist":
            if type(value) not in [np.ndarray, type(None)]:
                raise ValueError("Distortion coefficients must be a numpy array")
            if value is not None:
                if value.shape[0] != 5:
                    raise ValueError("Distortion coefficients must be a 5 element array")

        elif name == "units":
            if value not in ["mm", "m"]:
                raise ValueError("Units must be 'mm' or 'm'")
        elif name == "name":
            if type(value) not in [str, type(None)]:
                raise ValueError("Name must be a string")
        elif name == "visualization":
            if value not in ["both", "pyramid", "axes"]:
                raise ValueError("Visualization must be 'both', 'pyramid' or 'axis'")
        elif name == "camera_notation":
            if value not in ["opencv", "opengl"]:
                raise ValueError("Camera notation must be 'opencv' or 'opengl'")
        # TODO: Mauve add a Path type check from pathlib
        elif name == "image_path":
            if type(value) not in [str, type(None)]:
                raise ValueError("Image path must be a string")
        elif name == "depth_path":
            if type(value) not in [str, type(None)]:
                raise ValueError("Depth path must be a string")
        elif name == "depth_scale":
            if value <= 0:
                raise ValueError("Depth scale must be positive")
        elif name == "mask_path":
            if type(value) not in [str, type(None)]:
                raise ValueError("Mask path must be a string")
        elif name == "T_d2c":
            assert type(value) == np.ndarray
            if value.shape != (4, 4):
                raise ValueError("T_d2c must be a 4x4 matrix")

        elif name == "color":
            if type(value) not in [list, tuple, np.ndarray]:
                raise ValueError("Color must be a list or numpy array")
            if len(value) != 3:
                raise ValueError("Color must be a list of 3 elements")
            if not all([0 <= val <= 255 for val in value]):
                raise ValueError("Color values must be between 0 and 255")

        else:
            raise AttributeError(f"Attribute {name} not found")

        super().__setattr__(name, value)

    def __str__(self) -> str:
        print(self.Tmx, type(self.Tmx)) 
        angles = R.from_matrix(self.Tmx[:3, :3]).as_euler("zyx", degrees=True)
        angles = [round(angle, 2) for angle in angles]
        string = "Camera Pose:\n"
        if self.name:
            string += f"\tName: {self.name}\n"
        string += f"\tTranslation: {self.Tmx[:3, 3]} {self.units}\n\tRotation (ZYX): {angles} degrees\n\tAspect Ratio: {self.aspect_ratio}"
        return string

    def save(self, path: str) -> None:
        """saves the camera pose to a json file

        Args:
            path (str): path to save the camera pose
        """
        print(self.__dict__)
        d = self.__dict__
        for key in d:
            if type(d[key]) == np.ndarray:
                d[key] = d[key].tolist()

        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def load(self, path: str) -> None:
        """Loads the camera information from a json file 

        Args:
            path (str): path to the json file
        """
        with open(path) as f:
            d = json.load(f)

        for key in d:
            if type(d[key]) == list:
                d[key] = np.array(d[key])
                pass

        self.__dict__.update(d)

    def create_camera_pyramid(self, rgb:np.ndarray, cam_K:np.ndarray) -> None:
        """Based on the camera intrinsics and rgb image creates the camera pyramid

        Adds visualization of the camera pyramid to the scene with the notch
        depicting the top of the image
        
        Args:
            rgb (np.ndarray): RGB image [HxWx3] 
            cam_K (np.ndarray): Camera intrinsics matrix [3x3]
        """        
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultLit"
        h, w = rgb.shape[:2]
        camera_lineset = o3d.geometry.LineSet()
        camera_lineset = camera_lineset.create_camera_visualization(w,h,cam_K, np.eye(4), scale=0.1)
        camera_lineset.paint_uniform_color([.2, .2, .2])
        camera_points = np.asarray(camera_lineset.points)
        # self._scene.scene.add_geometry("camera", camera_lineset, mtl)


        # create top view of the pyramid
        left_top = camera_points[1,:]
        right_top = camera_points[2,:]
        right_bot = camera_points[3,:]
        left_bot = camera_points[4,:]
        # cam_img_plane = [left_top, right_top, right_bot, left_bot]
        # self._cam_img_plane = cam_img_plane
        height = np.abs(right_bot[1] - right_top[1])
        h = height * 0.33
        
        mid_width = (right_top[0] - left_top[0])/2

        notch_point = np.array([left_top[0] + mid_width, left_top[1] - h, left_top[2]])
        notch_points = np.array([left_top, right_top, notch_point])
        top_lineset = o3d.geometry.LineSet()
        top_lineset.points = o3d.utility.Vector3dVector(notch_points)
        top_lineset.lines = o3d.utility.Vector2iVector([[0, 2], [2,1]])
        top_lineset.paint_uniform_color([.2, .2, .2])
        # self._scene.scene.add_geometry("camera_top", top_lineset, mtl)

        vertices = np.array([left_top, right_top, right_bot, left_bot], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0,0], [1,1], [0,1]], dtype=np.float32)

    def create_image_plane(self, img:np.ndarray, left_top:np.ndarray, right_top:np.ndarray, right_bot:np.ndarray, left_bot:np.ndarray) -> None:
        """Creates the image plane with the texture of the image and adds it to the scene

        Args:
            img (np.ndarray): RGB image [HxWx3]
            left_top (np.ndarray): Top left corner of the image plane [x,y,z]
            right_top (np.ndarray): Top right corner of the image plane [x,y,z]
            right_bot (np.ndarray): Bottom right corner of the image plane [x,y,z]
            left_bot (np.ndarray): Bottom left corner of the image plane [x,y,z]
        """        
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        map_image = o3d.geometry.Image(img)
        
        # Create a rectangle mesh with the image as texture
        vertices = np.array([left_top, right_top, right_bot, left_bot], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0,0], [1,1], [0,1]], dtype=np.float32)


        # Create a material for the textured mesh
        material = o3d.visualization.rendering.MaterialRecord()
        material.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        material.base_metallic = 0.0
        material.base_roughness = 1.0
        material.albedo_img = map_image
        # Map the image to the camera pyramid
        texture_map = o3d.geometry.TriangleMesh()
        texture_map.vertices = o3d.utility.Vector3dVector(vertices)
        texture_map.triangles = o3d.utility.Vector3iVector(triangles)
        texture_map.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        # texture_map.textures = [map_image]

        # self._scene.scene.add_geometry("image_plane", texture_map, material)


if __name__ == "__main__":
    cp = CameraPose()
    cp.name = "Camera 2"

    cp.save("camera.json")
    print(cp)

    cp2 = CameraPose()
    cp2.load("camera.json")
    print(cp2)
