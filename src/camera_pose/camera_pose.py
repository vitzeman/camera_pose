from dataclasses import dataclass, field
from typing import List, Any
import json


import numpy as np
from scipy.spatial.transform import Rotation as R

M2MM = 1000.0
MM2M = 0.001


@dataclass
class CameraPose:
    """
    Data class to store the camera pose

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
    K: np.ndarray = None
    units: str = "m"
    dist: np.ndarray = None
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
        elif name == "K":
            if type(value) not in [np.ndarray, type(None)]:
                raise ValueError("Intrinsic matrix must be a numpy array")
            if value is not None:
                if value.shape != (3, 3):
                    raise ValueError("Intrinsic matrix must be a 3x3 matrix")
        elif name == "dist":
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
        angles = R.from_matrix(self.Tmx[:3, :3]).as_euler("zyx", degrees=True)
        angles = [round(angle, 2) for angle in angles]
        string = f"Camera Pose:\n"
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

        self.__dict__.update(d)


if __name__ == "__main__":
    cp = CameraPose()
    cp.name = "Camera 2"

    cp.save("camera.json")
    print(cp)

    cp2 = CameraPose()
    cp2.load("camera.json")
    print(cp2)
